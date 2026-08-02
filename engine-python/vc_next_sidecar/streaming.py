from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def equal_power_strengths(
    frames: int,
    *,
    fade_start_rate: float = 0.1,
    fade_end_rate: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Build complementary previous/current gains for one overlap region."""
    if frames < 2:
        raise ValueError("A crossfade requires at least two frames.")
    if not 0.0 <= fade_start_rate < fade_end_rate <= 1.0:
        raise ValueError("Crossfade rates must satisfy 0 <= start < end <= 1.")
    fade_start = int(frames * fade_start_rate)
    fade_end = max(fade_start + 1, int(frames * fade_end_rate))
    fade_end = min(frames, fade_end)
    progress = np.arange(fade_end - fade_start, dtype=np.float32) / (
        fade_end - fade_start
    )
    previous_curve = np.cos(progress * 0.5 * np.pi) ** 2
    current_curve = np.cos((1.0 - progress) * 0.5 * np.pi) ** 2
    previous = np.concatenate(
        (
            np.ones(fade_start, dtype=np.float32),
            previous_curve,
            np.zeros(frames - fade_end, dtype=np.float32),
        )
    )
    current = np.concatenate(
        (
            np.zeros(fade_start, dtype=np.float32),
            current_curve,
            np.ones(frames - fade_end, dtype=np.float32),
        )
    )
    return previous.astype(np.float32), current.astype(np.float32)


@dataclass(frozen=True)
class SolaResult:
    audio: np.ndarray
    offset_frames: int
    primed: bool


class SolaStitcher:
    """Stateful synchronized overlap-add for equally sized output hops.

    The candidate must contain ``search + hop + overlap`` frames. Its prefix
    contains the repeated tail context from the previous inference window.
    """

    def __init__(self, hop_frames: int, overlap_frames: int, search_frames: int) -> None:
        if hop_frames <= 0 or overlap_frames <= 1 or search_frames < 0:
            raise ValueError("Invalid SOLA frame configuration.")
        if overlap_frames >= hop_frames:
            raise ValueError("The SOLA overlap must be smaller than the output hop.")
        self.hop_frames = hop_frames
        self.overlap_frames = overlap_frames
        self.search_frames = search_frames
        self.previous_strength, self.current_strength = equal_power_strengths(
            overlap_frames
        )
        self._buffer: np.ndarray | None = None

    @property
    def candidate_frames(self) -> int:
        return self.search_frames + self.hop_frames + self.overlap_frames

    @property
    def primed(self) -> bool:
        return self._buffer is not None

    def reset(self) -> None:
        self._buffer = None

    def process(self, candidate: np.ndarray) -> SolaResult:
        audio = np.asarray(candidate, dtype=np.float32).reshape(-1)
        if audio.shape[0] != self.candidate_frames:
            raise ValueError(
                f"SOLA requires exactly {self.candidate_frames} candidate frames."
            )
        if not np.isfinite(audio).all():
            raise ValueError("SOLA received non-finite audio.")

        if self._buffer is None:
            self._buffer = audio[-self.overlap_frames :] * self.previous_strength
            return SolaResult(
                audio=np.zeros(self.hop_frames, dtype=np.float32),
                offset_frames=0,
                primed=False,
            )

        search_region = audio[: self.overlap_frames + self.search_frames]
        numerator = np.correlate(search_region, self._buffer, mode="valid")
        energy = np.convolve(
            np.square(search_region),
            np.ones(self.overlap_frames, dtype=np.float32),
            mode="valid",
        )
        scores = numerator / np.sqrt(energy + 1e-3)
        offset = int(np.argmax(scores))

        output = audio[offset : offset + self.hop_frames].copy()
        output[: self.overlap_frames] *= self.current_strength
        output[: self.overlap_frames] += self._buffer
        buffer_start = offset + self.hop_frames
        self._buffer = (
            audio[buffer_start : buffer_start + self.overlap_frames]
            * self.previous_strength
        )
        return SolaResult(audio=output, offset_frames=offset, primed=True)
