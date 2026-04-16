from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention


class DecoupledCrossAttnProcessor2_0(nn.Module):
    """
    Unified decoupled cross-attention processor for SDXL blocks.

    It expects ``encoder_hidden_states`` to be:
      [base_prompt_tokens ... | tax_token] or
      [base_prompt_tokens ... | tax_token | ref_token]

    and computes:
      Z = Attn(Q, K_base, V_base)
        + lambda_tax * Attn(Q, K_tax, V_tax)
        + lambda_ref * Attn(Q, K_ref, V_ref)
    """

    def __init__(self, attn: Attention) -> None:
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("DecoupledCrossAttnProcessor2_0 requires torch>=2.0")
        k_out, k_in = attn.to_k.weight.shape
        v_out, v_in = attn.to_v.weight.shape
        self.to_k_tax = nn.Linear(k_in, k_out, bias=False)
        self.to_v_tax = nn.Linear(v_in, v_out, bias=False)
        self.to_k_ref = nn.Linear(k_in, k_out, bias=False)
        self.to_v_ref = nn.Linear(v_in, v_out, bias=False)
        # Start from a near-zero perturbation for stability.
        nn.init.normal_(self.to_k_tax.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.to_v_tax.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.to_k_ref.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.to_v_ref.weight, mean=0.0, std=1e-3)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        lambda_tax: float = 1.0,
        lambda_ref: float = 1.0,
        use_reference_condition: bool = True,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        seq_len = encoder_hidden_states.shape[1]
        if use_reference_condition and seq_len >= 3:
            base_states = encoder_hidden_states[:, :-2, :]
            tax_states = encoder_hidden_states[:, -2:-1, :]
            ref_states = encoder_hidden_states[:, -1:, :]
        elif seq_len >= 2:
            base_states = encoder_hidden_states[:, :-1, :]
            tax_states = encoder_hidden_states[:, -1:, :]
            ref_states = None
        else:
            # Degenerate fallback: no extra tokens, treat the whole sequence as base.
            base_states = encoder_hidden_states
            tax_states = encoder_hidden_states[:, -1:, :]
            ref_states = None

        batch_size, base_seq_len, _ = base_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, base_seq_len, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        def _to_heads(x: torch.Tensor) -> torch.Tensor:
            inner_dim = x.shape[-1]
            head_dim = inner_dim // attn.heads
            return x.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query = _to_heads(query)
        if attn.norm_q is not None:
            query = attn.norm_q(query)

        key_base = _to_heads(attn.to_k(base_states))
        value_base = _to_heads(attn.to_v(base_states))
        key_tax = _to_heads(self.to_k_tax(tax_states))
        value_tax = _to_heads(self.to_v_tax(tax_states))
        if ref_states is not None:
            key_ref = _to_heads(self.to_k_ref(ref_states))
            value_ref = _to_heads(self.to_v_ref(ref_states))
        else:
            key_ref = None
            value_ref = None

        if attn.norm_k is not None:
            key_base = attn.norm_k(key_base)
            key_tax = attn.norm_k(key_tax)
            if key_ref is not None:
                key_ref = attn.norm_k(key_ref)

        out_base = F.scaled_dot_product_attention(query, key_base, value_base, attn_mask=attention_mask, dropout_p=0.0)
        out_tax = F.scaled_dot_product_attention(query, key_tax, value_tax, attn_mask=None, dropout_p=0.0)
        out_ref = (
            F.scaled_dot_product_attention(query, key_ref, value_ref, attn_mask=None, dropout_p=0.0)
            if key_ref is not None and value_ref is not None
            else torch.zeros_like(out_tax)
        )

        lambda_tax = float(lambda_tax)
        lambda_ref = float(lambda_ref)
        hidden_states = out_base + lambda_tax * out_tax + lambda_ref * out_ref

        head_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def attach_decoupled_cross_attention(unet: nn.Module) -> nn.ModuleList:
    """
    Replace all cross-attention processors in UNet with decoupled processors.
    Returns a ModuleList of newly created processors (trainable parameters).
    """
    new_processors: Dict[str, nn.Module] = {}
    created = nn.ModuleList()
    for name, proc in unet.attn_processors.items():
        module = unet.get_submodule(name.replace(".processor", ""))
        cross_dim = getattr(module, "cross_attention_dim", None)
        if cross_dim is None:
            new_processors[name] = proc
            continue
        dec = DecoupledCrossAttnProcessor2_0(module)
        new_processors[name] = dec
        created.append(dec)
    unet.set_attn_processor(new_processors)
    return created
