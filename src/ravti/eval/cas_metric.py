from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CASMetricStub:
    """
    CAS@1 placeholder: wire a torchvision / timm classifier checkpoint here.

    When `classifier` is None, returns a constant score so the evaluation stage
    still runs end-to-end in dry environments.
    """

    classifier: Optional[nn.Module] = None
    num_classes: int = 10_000

    @torch.inference_mode()
    def score(self, generated_batch: torch.Tensor, target_class_index: int) -> float:
        if self.classifier is None:
            return 0.0
        logits = self.classifier(generated_batch)
        pred = int(logits.argmax(dim=-1).item())
        return 1.0 if pred == target_class_index else 0.0
