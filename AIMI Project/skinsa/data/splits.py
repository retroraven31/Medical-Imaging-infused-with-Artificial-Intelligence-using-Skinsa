"""
Dataset splitting protocols:
1. --protocol paper: Stratified random 90/10 split evaluated on validation set (authors' approach).
2. --protocol lesion: StratifiedGroupKFold(10) grouped by lesion_id (80/10/10) to eliminate lesion leakage.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


def split_paper_protocol(
    df: pd.DataFrame,
    label_col: str = "dx",
    val_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baseline protocol: Stratified random 90/10 split on image level.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata dataframe containing image_id and dx columns.
    label_col : str
        Diagnosis column name.
    val_size : float
        Validation split fraction (default 0.10 for 90/10).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    train_df, val_df : Tuple[pd.DataFrame, pd.DataFrame]
    """
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state,
    )

    labels = df[label_col].values
    train_idx, val_idx = next(sss.split(df, labels))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    return train_df, val_df


def split_lesion_protocol(
    df: pd.DataFrame,
    label_col: str = "dx",
    group_col: str = "lesion_id",
    n_splits: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Enhanced protocol: StratifiedGroupKFold(10) grouped by lesion_id.
    Ensures images of the same patient lesion never land on both sides of the split.
    Creates an 80/10/10 split (8 train folds, 1 val fold, 1 test fold).

    Returns
    -------
    train_df, val_df, test_df : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    X = df.index.values
    y = df[label_col].values
    groups = df[group_col].values

    folds = list(sgkf.split(X, y, groups))
    # Fold 0 as test set (10%), Fold 1 as val set (10%), Folds 2..9 as train set (80%)
    test_idx = folds[0][1]
    val_idx = folds[1][1]
    # Train index is remaining indices
    all_indices = set(range(len(df)))
    train_idx = list(all_indices - set(test_idx) - set(val_idx))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def verify_no_lesion_leakage(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    group_col: str = "lesion_id",
) -> Tuple[bool, Set[str]]:
    """
    Verify zero lesion ID leakage across dataset splits.

    Returns
    -------
    is_clean : bool
        True if intersection is empty.
    leaked_lesions : set of str
        Set of overlapping lesion IDs (empty if clean).
    """
    train_lesions = set(train_df[group_col].dropna().unique())
    eval_lesions = set(eval_df[group_col].dropna().unique())
    leakage = train_lesions.intersection(eval_lesions)
    return len(leakage) == 0, leakage
