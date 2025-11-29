"""
Preprocessing pipeline for Ames Housing dataset
Phase A1: Standard Preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class AmesPreprocessor:
    """
    Comprehensive preprocessing for Ames Housing data

    Features:
    - Semantic imputation (NA meanings)
    - Grouped imputation (LotFrontage by Neighborhood)
    - Ordinal encoding for quality features
    - One-hot encoding for categorical features
    - Engineered features (TotalSF, HouseAge, etc.)
    """

    def __init__(self):
        self.numeric_impute_values = {}
        self.neighborhood_lot_frontage = {}
        self.feature_columns = None  # Store column names after fit
        self.fitted = False

    def fit_transform(self, df):
        """Fit preprocessor and transform training data"""
        df = df.copy()

        # 1. Create engineered features first
        df = self._create_engineered_features(df)

        # 2. Handle missing values
        df = self._handle_missing_values(df, fit=True)

        # 3. Encode quality features (ordinal)
        df = self._encode_quality_features(df)

        # 4. Encode categorical features (one-hot)
        df = self._encode_categorical_features(df)

        # Store column names for test set alignment
        self.feature_columns = df.columns.tolist()

        self.fitted = True
        return df

    def transform(self, df):
        """Transform test data using fitted parameters"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        df = df.copy()

        # Apply same transformations
        df = self._create_engineered_features(df)
        df = self._handle_missing_values(df, fit=False)
        df = self._encode_quality_features(df)
        df = self._encode_categorical_features(df)

        # Align columns with training set
        # Add missing columns (fill with 0)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Remove extra columns and reorder to match training
        df = df[self.feature_columns]

        return df

    def _create_engineered_features(self, df):
        """Create domain-specific engineered features"""

        # TotalSF: Total square footage
        df['TotalSF'] = (
            df['TotalBsmtSF'].fillna(0) +
            df['1stFlrSF'].fillna(0) +
            df['2ndFlrSF'].fillna(0)
        )

        # HouseAge: Age of house at sale
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']

        # RemodAge: Years since remodel
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

        # TotalBath: Total bathrooms (weighted)
        df['TotalBath'] = (
            df['FullBath'].fillna(0) +
            0.5 * df['HalfBath'].fillna(0) +
            df['BsmtFullBath'].fillna(0) +
            0.5 * df['BsmtHalfBath'].fillna(0)
        )

        # PorchArea: Total porch area
        df['PorchArea'] = (
            df['OpenPorchSF'].fillna(0) +
            df['EnclosedPorch'].fillna(0) +
            df['3SsnPorch'].fillna(0) +
            df['ScreenPorch'].fillna(0)
        )

        return df

    def _handle_missing_values(self, df, fit=False):
        """Handle missing values with semantic and grouped imputation"""

        # Features where NA means "None" or absence
        na_means_none = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'MasVnrType'
        ]

        for col in na_means_none:
            if col in df.columns:
                df[col] = df[col].fillna('None')

        # Numeric features where NA means 0
        na_means_zero = [
            'GarageYrBlt', 'GarageArea', 'GarageCars',
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
        ]

        for col in na_means_zero:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # LotFrontage: Grouped imputation by Neighborhood
        if 'LotFrontage' in df.columns:
            if fit:
                # Calculate median LotFrontage for each Neighborhood
                self.neighborhood_lot_frontage = df.groupby('Neighborhood')['LotFrontage'].median().to_dict()

            # Fill missing LotFrontage based on Neighborhood median
            df['LotFrontage'] = df.apply(
                lambda row: self.neighborhood_lot_frontage.get(row['Neighborhood'], 0)
                if pd.isna(row['LotFrontage']) else row['LotFrontage'],
                axis=1
            )

        # Remaining numeric features: median imputation
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        if fit:
            for col in numeric_cols:
                if df[col].isna().sum() > 0:
                    self.numeric_impute_values[col] = df[col].median()

        for col, val in self.numeric_impute_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        # Remaining categorical features: mode imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'None')

        return df

    def _encode_quality_features(self, df):
        """Ordinal encoding for quality features"""

        # Quality mapping: Excellent to Poor
        qual_map = {
            'Ex': 5,  # Excellent
            'Gd': 4,  # Good
            'TA': 3,  # Typical/Average
            'Fa': 2,  # Fair
            'Po': 1,  # Poor
            'None': 0  # Not present
        }

        # Quality features to encode
        qual_features = [
            'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
            'HeatingQC', 'KitchenQual', 'FireplaceQu',
            'GarageQual', 'GarageCond', 'PoolQC'
        ]

        for col in qual_features:
            if col in df.columns:
                df[col] = df[col].map(qual_map).fillna(0).astype(int)

        # Basement Exposure: ordinal
        if 'BsmtExposure' in df.columns:
            bsmt_exp_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
            df['BsmtExposure'] = df['BsmtExposure'].map(bsmt_exp_map).fillna(0).astype(int)

        # Basement Finish Type: ordinal
        bsmt_fin_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}

        if 'BsmtFinType1' in df.columns:
            df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmt_fin_map).fillna(0).astype(int)

        if 'BsmtFinType2' in df.columns:
            df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmt_fin_map).fillna(0).astype(int)

        # Functional: ordinal
        if 'Functional' in df.columns:
            func_map = {
                'Typ': 7,  # Typical Functionality
                'Min1': 6, 'Min2': 5,  # Minor Deductions
                'Mod': 4,  # Moderate Deductions
                'Maj1': 3, 'Maj2': 2,  # Major Deductions
                'Sev': 1,  # Severely Damaged
                'Sal': 0   # Salvage only
            }
            df['Functional'] = df['Functional'].map(func_map).fillna(7).astype(int)

        # Garage Finish: ordinal
        if 'GarageFinish' in df.columns:
            garage_fin_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}
            df['GarageFinish'] = df['GarageFinish'].map(garage_fin_map).fillna(0).astype(int)

        return df

    def _encode_categorical_features(self, df):
        """One-hot encoding for categorical features"""

        # Get remaining categorical columns (after ordinal encoding)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        if len(categorical_cols) > 0:
            # One-hot encode
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        return df
