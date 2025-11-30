"""
Smart Minimal Preprocessor for the Feature Engineering Agent.

This preprocessor balances minimal feature engineering with smart data handling:
1. Semantic NA handling (missing pool = no pool, not "unknown")
2. Ordinal encoding for quality features (Ex=5 > Gd=4 > TA=3 > Fa=2 > Po=1)
3. Label encoding for nominal categories
4. Smart imputation (neighborhood median for LotFrontage)

NO feature engineering (TotalSF, HouseAge, etc.) - agent discovers these.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


# Quality mappings - preserve ordinal meaning
QUALITY_MAP = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, None: 0, np.nan: 0}
EXPOSURE_MAP = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0, None: 0, np.nan: 0}
BSMT_FIN_MAP = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0, None: 0, np.nan: 0}
GARAGE_FIN_MAP = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0, None: 0, np.nan: 0}
FUNCTIONAL_MAP = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}

# Columns where NA means "none/absent" (not missing data)
NA_MEANS_NONE = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]

# Numeric columns where NA means zero
NA_MEANS_ZERO = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]

# Quality columns to ordinal encode
QUALITY_COLS = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'FireplaceQu',
                'GarageQual', 'GarageCond', 'PoolQC']


class MinimalPreprocessor:
    """Smart minimal preprocessing: semantic NA + ordinal quality + label encode"""

    def __init__(self):
        self.numeric_medians: Dict[str, float] = {}
        self.categorical_modes: Dict[str, str] = {}
        self.label_encoders: Dict[str, Dict[str, int]] = {}
        self.neighborhood_lot_frontage: Dict[str, float] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> 'MinimalPreprocessor':
        """Learn imputation values and encodings from training data."""

        # Learn neighborhood-specific LotFrontage medians
        if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
            self.neighborhood_lot_frontage = df.groupby('Neighborhood')['LotFrontage'].median().to_dict()

        # Learn medians for numeric columns (excluding special cases)
        for col in df.select_dtypes(include=['number']).columns:
            if col not in ['Id', 'SalePrice'] + NA_MEANS_ZERO:
                self.numeric_medians[col] = df[col].median()

        # Learn modes and label encodings for categorical columns
        # Exclude quality columns (they get ordinal encoding)
        ordinal_cols = set(QUALITY_COLS + ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                                            'GarageFinish', 'Functional'])

        for col in df.select_dtypes(include=['object']).columns:
            if col in ordinal_cols:
                continue  # Skip - will be ordinal encoded

            # Get mode for imputation (after filling NA_MEANS_NONE with 'None')
            col_data = df[col].copy()
            if col in NA_MEANS_NONE:
                col_data = col_data.fillna('None')

            mode_series = col_data.mode()
            self.categorical_modes[col] = mode_series[0] if not mode_series.empty else 'Unknown'

            # Create label encoding map (sorted for consistency)
            unique_vals = col_data.dropna().unique()
            if col in NA_MEANS_NONE:
                unique_vals = list(unique_vals) + ['None']
            unique_vals = sorted(set(unique_vals))
            self.label_encoders[col] = {v: i for i, v in enumerate(unique_vals)}

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply semantic imputation, ordinal encoding, and label encoding."""
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        df = df.copy()

        # 1. Semantic NA handling - fill with 'None' for absence
        for col in NA_MEANS_NONE:
            if col in df.columns:
                df[col] = df[col].fillna('None')

        # 2. Numeric NA = 0 (no garage = 0 garage area, etc.)
        for col in NA_MEANS_ZERO:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # 3. LotFrontage - neighborhood median
        if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
            for idx in df[df['LotFrontage'].isna()].index:
                neighborhood = df.loc[idx, 'Neighborhood']
                df.loc[idx, 'LotFrontage'] = self.neighborhood_lot_frontage.get(
                    neighborhood, df['LotFrontage'].median()
                )

        # 4. Other numeric - median imputation
        for col, median in self.numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)

        # 5. Ordinal encode quality features
        for col in QUALITY_COLS:
            if col in df.columns:
                df[col] = df[col].map(QUALITY_MAP).fillna(0).astype(int)

        # 6. Ordinal encode other special columns
        if 'BsmtExposure' in df.columns:
            df['BsmtExposure'] = df['BsmtExposure'].map(EXPOSURE_MAP).fillna(0).astype(int)

        for col in ['BsmtFinType1', 'BsmtFinType2']:
            if col in df.columns:
                df[col] = df[col].map(BSMT_FIN_MAP).fillna(0).astype(int)

        if 'GarageFinish' in df.columns:
            df['GarageFinish'] = df['GarageFinish'].map(GARAGE_FIN_MAP).fillna(0).astype(int)

        if 'Functional' in df.columns:
            df['Functional'] = df['Functional'].map(FUNCTIONAL_MAP).fillna(7).astype(int)

        # 7. Label encode remaining categorical columns
        for col, mode in self.categorical_modes.items():
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].fillna(mode)
                encoder = self.label_encoders.get(col, {})
                df[col] = df[col].map(lambda x: encoder.get(x, -1))

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def get_feature_info(self) -> Dict[str, Any]:
        """Return information about the preprocessing for debugging."""
        return {
            'n_numeric_features': len(self.numeric_medians),
            'n_categorical_features': len(self.categorical_modes),
            'n_ordinal_encoded': len(QUALITY_COLS) + 4,  # quality + special cols
            'semantic_na_cols': len(NA_MEANS_NONE),
        }
