"""
True Minimal Preprocessor for the Feature Engineering Agent.

This preprocessor does the absolute minimum to make data usable:
1. Numeric NaN → 0 (agent can detect absence via > 0)
2. Categorical NaN → "None" string
3. Convert categoricals to pd.Categorical dtype

Agent discovers EVERYTHING else:
- Ordinal encoding (from data_description.txt)
- Binary presence features (HasGarage, HasPool)
- Grouped features (neighborhood-based)
- Engineered features (TotalSF, HouseAge, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


class MinimalPreprocessor:
    """
    True minimal preprocessing - agent discovers everything.

    Does ONLY:
    1. Numeric NaN → 0 (agent can detect absence via > 0)
    2. Categorical NaN → "None" string
    3. Convert categoricals to pd.Categorical dtype

    Does NOT:
    - Ordinal encoding (agent discovers from data_description)
    - Grouped imputation (agent discovers)
    - Feature engineering (agent discovers)
    """

    def __init__(self):
        self.categories: Dict[str, List[str]] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> 'MinimalPreprocessor':
        """Learn categories for each categorical column."""
        for col in df.select_dtypes(include=['object']).columns:
            # Get all unique values plus 'None' for missing
            unique_vals = list(df[col].dropna().unique())
            if 'None' not in unique_vals:
                unique_vals.append('None')
            self.categories[col] = sorted(set(unique_vals))

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply minimal transformation."""
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        df = df.copy()

        # 1. Numeric NaN → 0 (allows agent to detect absence via > 0)
        for col in df.select_dtypes(include=['number']).columns:
            if col not in ['Id', 'SalePrice']:
                df[col] = df[col].fillna(0)

        # 2. Categorical NaN → "None", convert to Categorical dtype
        for col, cats in self.categories.items():
            if col in df.columns:
                df[col] = df[col].fillna('None')
                df[col] = pd.Categorical(df[col], categories=cats)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def get_feature_info(self) -> Dict[str, Any]:
        """Return information about the preprocessing for debugging."""
        return {
            'n_categorical_features': len(self.categories),
            'categorical_features': list(self.categories.keys()),
            'total_categories': sum(len(cats) for cats in self.categories.values()),
        }
