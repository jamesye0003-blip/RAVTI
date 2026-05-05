"""Classification-accuracy metrics: CAS@1 and CAS@k (e.g. CAS@5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ClassificationAccuracyMetric:
    """Top-k accuracy against a target class index (CAS@k)."""

    classifier: Optional[nn.Module] = None
    num_classes: int = 10_000
    image_size: int = 224
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    class_index_remap: dict[int, int] | None = None

    @classmethod
    def from_config(cls, cfg: dict, device: torch.device) -> "ClassificationAccuracyMetric":
        """Load the classifier from the configuration"""
        ckpt = cfg.get("cas_classifier")
        if not ckpt:
            return cls(classifier=None)
        p = Path(str(ckpt))
        if not p.is_absolute():
            from ravti.paths import project_root

            p = (project_root() / p).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"evaluation.cas_classifier not found: {p}")

        model_name = str(cfg.get("cas_classifier_arch", "resnet50")).strip()
        image_size = int(cfg.get("cas_image_size", 224))
        num_classes = int(cfg.get("cas_num_classes", 10_000))

        payload = torch.load(p, map_location="cpu")
        state = payload
        if isinstance(payload, dict):
            for k in ("state_dict", "model_state_dict", "classifier_state_dict"):
                if k in payload and isinstance(payload[k], dict):
                    state = payload[k]
                    break
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported CAS checkpoint format: {p}")

        # Prefer checkpoint-declared class count to avoid fc size mismatch.
        if isinstance(payload, dict):
            if payload.get("num_classes") is not None:
                try:
                    num_classes = int(payload["num_classes"])
                except Exception:
                    pass
        if "fc.weight" in state and hasattr(state["fc.weight"], "shape"):
            try:
                num_classes = int(state["fc.weight"].shape[0])
            except Exception:
                pass

        try:
            from torchvision import models as tvm
        except Exception as e:
            raise RuntimeError("torchvision is required for CAS classifier loading") from e
        if not hasattr(tvm, model_name):
            raise ValueError(f"Unsupported evaluation.cas_classifier_arch={model_name!r} in torchvision.models")
        model_ctor = getattr(tvm, model_name)
        model = model_ctor(weights=None)
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        class_index_remap: dict[int, int] | None = None
        if isinstance(payload, dict):
            raw_map = payload.get("class_index_remap")
            if isinstance(raw_map, dict):
                class_index_remap = {}
                for k, v in raw_map.items():
                    try:
                        class_index_remap[int(k)] = int(v)
                    except Exception:
                        continue
        cleaned = {}
        for k, v in state.items():
            kk = str(k)
            if kk.startswith("module."):
                kk = kk[len("module.") :]
            cleaned[kk] = v
        model.load_state_dict(cleaned, strict=False)
        model.to(device)
        model.eval()
        return cls(
            classifier=model,
            num_classes=num_classes,
            image_size=image_size,
            class_index_remap=class_index_remap,
        )

    def _preprocess(self, generated_batch: torch.Tensor) -> torch.Tensor:
        """Preprocess the generated batch to the input size of the classifier"""
        x = generated_batch
        if x.ndim == 3:
            x = x.unsqueeze(0)
        
        # Clamp the generated batch to the range [0, 1]
        x = torch.clamp(x.float(), 0.0, 1.0)

        # Interpolate the generated batch to the input size of the classifier
        x = torch.nn.functional.interpolate(
            x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )

        # Normalize the generated batch
        mean = torch.tensor(self.mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.inference_mode()
    def cas_at_k(self, generated_batch: torch.Tensor, target_class_index: int | None, k: int) -> float:
        # If the classifier is not loaded or the target class index is not provided, return 0
        if self.classifier is None or target_class_index is None:
            return 0.0

        # Ensure the k value is at least 1
        k = max(1, int(k))

        # Preprocess the generated batch
        xb = self._preprocess(generated_batch).to(next(self.classifier.parameters()).device)
        logits = self.classifier(xb)
        n_cls = int(logits.shape[-1])
        k_eff = min(k, n_cls)
        _, topk = logits.topk(k_eff, dim=-1)
        t = int(target_class_index)
        if self.class_index_remap is not None:
            mapped = self.class_index_remap.get(t)
            if mapped is None:
                return 0.0
            t = int(mapped)
        return float((topk == t).any(dim=-1).item())
