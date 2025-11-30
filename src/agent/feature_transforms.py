"""
Feature transforms for improving model performance

Based on techniques from top Kaggle solutions:
- Outlier removal
- Skewness correction (BoxCox transform)
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p


def remove_outliers(df: pd.DataFrame, target_col: str = 'SalePrice') -> pd.DataFrame:
    """
    Remove known outliers from Ames Housing dataset.

    Based on EDA from top Kaggle solutions:
    - Houses with GrLivArea > 4500 are outliers (very large houses sold cheaply)

    Args:
        df: Input dataframe
        target_col: Name of target column (only remove if target exists, i.e., training data)

    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    initial_rows = len(df)

    # Only remove outliers from training data (where target exists)
    if target_col in df.columns:
        # GrLivArea outliers: very large houses with low prices
        if 'GrLivArea' in df.columns:
            mask = df['GrLivArea'] < 4500
            df = df[mask]

    removed = initial_rows - len(df)
    if removed > 0:
        print(f"   Removed {removed} outliers ({removed/initial_rows*100:.1f}%)")

    return df


def fix_skewness(
    df: pd.DataFrame,
    threshold: float = 0.75,
    lam: float = 0.15,
    exclude_cols: list = None
) -> pd.DataFrame:
    """
    Apply Box-Cox transform to highly skewed numeric features.

    This helps linear models and can improve tree-based models too.
    Uses boxcox1p which handles zeros: boxcox1p(x, lam) = boxcox(x + 1, lam)

    Args:
        df: Input dataframe
        threshold: Skewness threshold (features with |skewness| > threshold are transformed)
        lam: Lambda parameter for Box-Cox transform (0.15 from top solutions)
        exclude_cols: Columns to exclude from transformation

    Returns:
        DataFrame with skewed features transformed
    """
    df = df.copy()

    if exclude_cols is None:
        exclude_cols = ['Id', 'SalePrice']

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Calculate skewness for each numeric column
    skewness = df[numeric_cols].apply(lambda x: skew(x.dropna()))

    # Find highly skewed features
    skewed_features = skewness[abs(skewness) > threshold].index.tolist()

    if skewed_features:
        print(f"   Applying Box-Cox transform to {len(skewed_features)} skewed features")

        for col in skewed_features:
            try:
                # Ensure non-negative values for boxcox1p
                min_val = df[col].min()
                if min_val < 0:
                    # Shift to make non-negative
                    df[col] = df[col] - min_val

                df[col] = boxcox1p(df[col], lam)
            except Exception as e:
                print(f"      Warning: Could not transform {col}: {e}")

    return df


def preprocess_for_agent(
    df: pd.DataFrame,
    remove_outliers_flag: bool = True,
    fix_skewness_flag: bool = True,
    target_col: str = 'SalePrice'
) -> pd.DataFrame:
    """
    Apply all preprocessing steps for the feature engineering agent.

    Args:
        df: Input dataframe (after basic preprocessing)
        remove_outliers_flag: Whether to remove outliers
        fix_skewness_flag: Whether to fix skewness
        target_col: Name of target column

    Returns:
        Preprocessed DataFrame
    """
    print("   Applying advanced preprocessing...")

    if remove_outliers_flag:
        df = remove_outliers(df, target_col=target_col)

    if fix_skewness_flag:
        df = fix_skewness(df, exclude_cols=['Id', target_col])

    return df
