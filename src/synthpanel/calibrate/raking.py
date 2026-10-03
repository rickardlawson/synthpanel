"""Raking / iterative proportional fitting (IPF) av agentvekter.

Justerer vektene slik at vektede summer treffer kjente marginaler
(f.eks. SSB-tall), mens vektene endres så lite som mulig.
Dette er samme metode som brukes til å vekte vanlige meningsmålinger.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RakeResult:
    weights: np.ndarray
    iterations: int
    max_misallocated: float
    converged: bool
    dropped_share: dict[str, float]
    misallocated: dict[str, float]


def rake(
    df: pd.DataFrame,
    margins: dict[str | tuple[str, ...], pd.Series],
    base_weights: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-3,
    bounds: tuple[float, float] | None = None,
    max_missing_share: float = 0.001,
) -> RakeResult:
    """Merk: marginalene tilpasses i rekkefølge, så den siste i `margins`
    treffes eksakt etter hver runde. Legg den viktigste sist."""
    """Kalibrer vekter mot én eller flere marginaler.

    margins: nøkkel er en kolonne eller en tuple av kolonner; verdien er en
    Series indeksert på kategoriene (MultiIndex for tuple) med måltotaler.
    bounds: valgfri (lav, høy) grense for vekt / basisvekt, for å unngå
    ekstreme vekter.
    """
    n = len(df)
    w0 = np.ones(n) if base_weights is None else np.asarray(base_weights, dtype=float).copy()
    w = w0.copy()

    groups = []
    dropped: dict[str, float] = {}
    for key, target in margins.items():
        cols = [key] if isinstance(key, str) else list(key)
        keys = df[cols[0]] if len(cols) == 1 else pd.MultiIndex.from_frame(df[cols])
        codes, uniques = pd.factorize(keys)
        tgt = target.reindex(uniques).to_numpy(dtype=float)
        if np.isnan(tgt).any():
            missing = [u for u, t in zip(uniques, tgt) if np.isnan(t)]
            raise ValueError(f"Margin {key} mangler mål for kategoriene {missing[:5]}")
        # Kategorier med mål men uten agenter (sjeldne kombinasjoner): masse
        # fordeles proporsjonalt på resten slik at totalen bevares.
        missing_mass = float(target.sum() - tgt.sum())
        share = missing_mass / float(target.sum()) if target.sum() else 0.0
        if share > max_missing_share:
            raise ValueError(
                f"Margin {key}: {share:.2%} av målet ligger i kategorier uten agenter"
            )
        if missing_mass > 0:
            tgt = tgt * float(target.sum()) / tgt.sum()
        dropped[str(key)] = share
        groups.append((codes, tgt, len(uniques)))

    max_err = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        for codes, tgt, k in groups:
            current = np.bincount(codes, weights=w, minlength=k)
            factor = np.divide(tgt, current, out=np.ones_like(tgt), where=current > 0)
            w *= factor[codes]
            if bounds is not None:
                w = np.clip(w, w0 * bounds[0], w0 * bounds[1])
        errors = {}
        for (codes, tgt, k), key in zip(groups, margins):
            current = np.bincount(codes, weights=w, minlength=k)
            # Andel av befolkningen som ligger i feil kategori (0 = perfekt).
            errors[str(key)] = float(0.5 * np.abs(current - tgt).sum() / tgt.sum())
        max_err = max(errors.values())
        if max_err < tol:
            break
    return RakeResult(weights=w, iterations=it, max_misallocated=max_err,
                      converged=max_err < tol, dropped_share=dropped,
                      misallocated=errors)


def effective_sample_size(w: np.ndarray) -> float:
    """Kish' effektive utvalgsstørrelse – faller når vektene blir ujevne."""
    return float(w.sum() ** 2 / (w**2).sum())
