from __future__ import annotations

import random
from enum import Enum


class OffloadTarget(str, Enum):
    LOCAL = "LOCAL"
    FOG = "FOG"
    CLOUD = "CLOUD"


class RandomOffloader:
    name = "Random"

    def __init__(self, seed: int = 11):
        self._rng = random.Random(seed)
        self._targets = tuple(OffloadTarget)

    def choose_target(self) -> OffloadTarget:
        return self._rng.choice(self._targets)

    def packet_lost(self, packet_loss_percent: float) -> bool:
        bounded = max(0.0, min(packet_loss_percent, 100.0))
        return self._rng.random() < (bounded / 100.0)

    def transmit_with_retries(
        self,
        packet_loss_percent: float,
        max_retransmissions: int,
    ) -> tuple[bool, int, int]:
        max_attempts = 1 + max(0, max_retransmissions)
        for attempt in range(1, max_attempts + 1):
            if not self.packet_lost(packet_loss_percent):
                return False, attempt, attempt - 1
        return True, max_attempts, max_attempts - 1
