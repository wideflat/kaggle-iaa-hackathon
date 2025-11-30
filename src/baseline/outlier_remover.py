"""
Outlier removal for Ames Housing dataset

Based on well-documented outliers from top Kaggle solutions.
These are houses with GrLivArea > 4000 sq ft but SalePrice < $300,000,
which are clear anomalies (likely partial sales or data errors).
"""

import pandas as pd


class OutlierRemover:
    """
    Remove known outliers from Ames Housing training data.

    The Ames Housing dataset has 2 well-known outliers that are
    consistently removed by top Kaggle solutions:
    - Houses with GrLivArea > 4000 sq ft
    - But SalePrice < $300,000

    These represent partial sales or recording errors and negatively
    impact model training.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize outlier remover.

        Args:
            verbose: Print removal information
        """
        self.verbose = verbose
        self.removed_indices = []

    def remove_known_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the 2 famous Ames Housing outliers.

        Only removes from training data (requires SalePrice column).

        Args:
            df: DataFrame with housing data

        Returns:
            DataFrame with outliers removed
        """
        # Only remove from training data
        if 'SalePrice' not in df.columns:
            if self.verbose:
                print("   Skipping outlier removal (no SalePrice - test data)")
            return df

        # Identify outliers: large houses with suspiciously low prices
        mask = (df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)
        self.removed_indices = df[mask].index.tolist()

        removed_count = mask.sum()
        if removed_count > 0 and self.verbose:
            print(f"   Removed {removed_count} outlier(s):")
            for idx in self.removed_indices:
                row = df.loc[idx]
                print(f"      - Index {idx}: GrLivArea={row['GrLivArea']}, "
                      f"SalePrice=${row['SalePrice']:,.0f}")

        return df[~mask].reset_index(drop=True)

    def remove_by_iqr(
        self,
        df: pd.DataFrame,
        column: str,
        multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers using IQR method (optional).

        Args:
            df: DataFrame
            column: Column to check for outliers
            multiplier: IQR multiplier (default 1.5 = standard)

        Returns:
            DataFrame with outliers removed
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)

        if self.verbose:
            removed = (~mask).sum()
            if removed > 0:
                print(f"   IQR removal on '{column}': {removed} outliers")

        return df[mask].reset_index(drop=True)

    def get_removed_indices(self) -> list:
        """Get indices of removed outliers from last call."""
        return self.removed_indices
