"""RVC-compatible sample-rate conversion.

w-okada's RVCv2 implementation uses :mod:`resampy`'s precomputed
``kaiser_fast`` filter at the 48 kHz device boundary.  Keeping that choice in
one small adapter makes the live and offline paths use the same filter.  The
torchaudio fallback preserves a usable installation when an older runtime did
not install resampy yet; the setup requirements install the exact path.
"""

from __future__ import annotations

from typing import Any

import numpy as np


KAISER_FAST_ROLLOFF = 0.8682120388377784
KAISER_FAST_BETA = 9.90322
KAISER_FAST_WIDTH = 24


def resample_kaiser_fast(
    samples: Any,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample a mono float waveform with w-okada's RVC filter contract.

    ``resampy`` is intentionally imported lazily because the UI can still
    start in a diagnostic-only environment before the full Python runtime is
    installed.  The fallback uses torchaudio's Kaiser sinc kernel with the
    same published fast-filter parameters rather than its default Hann kernel.
    """

    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if source_rate == target_rate:
        return np.ascontiguousarray(values.copy(), dtype=np.float32)

    try:
        import resampy
    except ImportError:
        import torch
        import torchaudio.functional as audio_functional

        converted = audio_functional.resample(
            torch.from_numpy(values),
            source_rate,
            target_rate,
            lowpass_filter_width=KAISER_FAST_WIDTH,
            rolloff=KAISER_FAST_ROLLOFF,
            resampling_method="sinc_interp_kaiser",
            beta=KAISER_FAST_BETA,
        ).numpy()
    else:
        converted = resampy.resample(
            values,
            source_rate,
            target_rate,
            filter="kaiser_fast",
        )
    return np.ascontiguousarray(np.asarray(converted, dtype=np.float32))
