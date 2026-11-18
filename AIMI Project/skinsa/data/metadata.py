"""
Metadata processing and encoding module for skinsa.
Encodes patient age, sex, and anatomical localization into an exact 21-feature vector:
- Sex: 3 features (male, female, unknown)
- Localization: 15 features
- Age: 3 features (normalized age, age missing indicator, non-linear age term)
Total = 21 features.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from skinsa.config import (
    METADATA_LOCALIZATION_CATEGORIES,
    METADATA_SEX_CATEGORIES,
    NUM_METADATA_FEATURES,
)


def encode_metadata_dict(
    meta: Dict[str, Union[float, int, str]],
) -> np.ndarray:
    """
    Encode a single patient's metadata dictionary into a 21-dimensional vector.

    Parameters
    ----------
    meta : dict
        Keys: 'age' (numeric or None), 'sex' (str), 'localization' (str).

    Returns
    -------
    vec : np.ndarray of shape (21,)
    """
    vec = np.zeros(NUM_METADATA_FEATURES, dtype=np.float32)

    # 1. Sex encoding (3 features: male, female, unknown)
    sex_str = str(meta.get("sex", "unknown")).strip().lower()
    for i, cat in enumerate(METADATA_SEX_CATEGORIES):
        if sex_str == cat:
            vec[i] = 1.0
            break
    else:
        # Fallback to 'unknown'
        vec[METADATA_SEX_CATEGORIES.index("unknown")] = 1.0

    offset = len(METADATA_SEX_CATEGORIES)  # 3

    # 2. Localization encoding (15 features)
    loc_str = str(meta.get("localization", "unknown")).strip().lower()
    for j, cat in enumerate(METADATA_LOCALIZATION_CATEGORIES):
        if loc_str == cat:
            vec[offset + j] = 1.0
            break
    else:
        vec[offset + METADATA_LOCALIZATION_CATEGORIES.index("unknown")] = 1.0

    offset += len(METADATA_LOCALIZATION_CATEGORIES)  # 3 + 15 = 18

    # 3. Age encoding (3 features: normalized, missing_flag, age^2)
    age_val = meta.get("age", None)
    if age_val is None or pd.isna(age_val) or age_val == "" or age_val < 0:
        # Impute mean age ~50.0
        vec[offset] = 0.50
        vec[offset + 1] = 1.0  # missing flag
        vec[offset + 2] = 0.25
    else:
        norm_age = float(age_val) / 100.0
        vec[offset] = np.clip(norm_age, 0.0, 1.2)
        vec[offset + 1] = 0.0  # present flag
        vec[offset + 2] = np.clip(norm_age ** 2, 0.0, 1.44)

    return vec


def encode_metadata_dataframe(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Batch encode a DataFrame containing columns 'age', 'sex', 'localization'.

    Returns
    -------
    X_meta : np.ndarray of shape (N, 21)
    """
    n = len(df)
    matrix = np.zeros((n, NUM_METADATA_FEATURES), dtype=np.float32)

    for idx, (_, row) in enumerate(df.iterrows()):
        matrix[idx] = encode_metadata_dict({
            "age": row.get("age", None),
            "sex": row.get("sex", "unknown"),
            "localization": row.get("localization", "unknown"),
        })

    return matrix
