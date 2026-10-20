
from __future__ import annotations

from app.models.softmax_head import SoftmaxClassificationHead


class TargetHead(SoftmaxClassificationHead):
    """Softmax head over the configured target categories."""
