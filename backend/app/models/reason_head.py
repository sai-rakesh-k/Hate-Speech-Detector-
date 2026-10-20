
from __future__ import annotations

from app.models.softmax_head import SoftmaxClassificationHead


class ReasonHead(SoftmaxClassificationHead):
    """Softmax head over the configured reason categories."""

