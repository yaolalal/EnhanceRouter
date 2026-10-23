"""Isolated cross-domain Router B/C implementation.

This package intentionally does not import or mutate the legacy V4/V5 router.
"""

from .models import DirectPreferenceRouter, FactorizedOutcomeRouter

__all__ = ["DirectPreferenceRouter", "FactorizedOutcomeRouter"]

