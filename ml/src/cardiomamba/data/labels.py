"""PTB-XL 5-class diagnostic superclass label mapping (`labels-superdiagnostic-v1`).

Rule: every SCP statement in a record's `scp_codes` whose row in `scp_statements.csv` has
`diagnostic == 1` is mapped through `diagnostic_class`; the record target is the multi-hot
union over CLASS_NAMES. Likelihood values are ignored (PTB-XL benchmark convention).
Form/rhythm-only statements do not contribute.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

LABEL_MAPPING_VERSION = "labels-superdiagnostic-v1"
CLASS_NAMES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")
NUM_CLASSES = len(CLASS_NAMES)


def parse_scp_codes(raw: str | Mapping[str, float]) -> dict[str, float]:
    """Parse the `scp_codes` column value (a Python-literal dict string) into a dict."""
    if isinstance(raw, Mapping):
        return dict(raw)
    value = ast.literal_eval(raw)
    if not isinstance(value, dict):
        raise TypeError(f"scp_codes is not a dict literal: {raw!r}")
    return value


def build_code_to_superclass(scp_statements: pd.DataFrame) -> dict[str, str]:
    """Map each diagnostic SCP code (index of scp_statements.csv) to its superclass."""
    diag = scp_statements[scp_statements["diagnostic"] == 1]
    mapping = diag["diagnostic_class"].to_dict()
    unknown = {c for c in mapping.values() if c not in CLASS_NAMES}
    if unknown or diag["diagnostic_class"].isna().any():
        raise ValueError(f"Unexpected diagnostic_class values: {unknown or 'NaN present'}")
    return mapping


def superclasses_for(codes: Iterable[str], code_to_superclass: Mapping[str, str],
                     known_codes: Iterable[str] | None = None) -> set[str]:
    """Return the set of superclasses implied by `codes` (likelihoods are not consulted).

    If `known_codes` is given, any code outside it raises ValueError (guards against
    metadata/statement-table mismatches instead of silently dropping labels).
    """
    codes = list(codes)
    if known_codes is not None:
        unknown = set(codes) - set(known_codes)
        if unknown:
            raise ValueError(f"SCP codes not present in scp_statements.csv: {sorted(unknown)}")
    return {code_to_superclass[c] for c in codes if c in code_to_superclass}


def multi_hot(superclasses: Iterable[str]) -> np.ndarray:
    """Encode a set of superclass names as float32 [NUM_CLASSES] in CLASS_NAMES order."""
    present = set(superclasses)
    return np.array([c in present for c in CLASS_NAMES], dtype=np.float32)
