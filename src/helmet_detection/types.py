from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    cls: int
    conf: float
    label: str
    track_id: int | None = None
    is_violation: bool = False

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2
