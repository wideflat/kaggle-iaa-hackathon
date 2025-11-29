"""
Safe code executor for LLM-generated feature engineering code
"""

import pandas as pd
import numpy as np
import traceback
from typing import Tuple, Optional


class CodeExecutor:
    """
    Safely execute LLM-generated Python code for feature engineering

    Features:
    - Sandboxed execution environment
    - Only allows pandas/numpy operations
    - Validates output dataframe
    - Captures and reports errors
    """

    def __init__(self):
        """Initialize executor with safe globals"""
        # Define safe globals - only pandas and numpy
        self.safe_globals = {
            'pd': pd,
            'np': np,
            # Restrict builtins to prevent dangerous operations
            '__builtins__': {
                'len': len,
                'range': range,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'round': round,
                'print': print,  # Allow print for debugging
            }
        }

    def execute(
        self,
        code: str,
        df: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute generated code on a dataframe

        Args:
            code: Python code to execute (should modify 'df')
            df: Input dataframe to transform

        Returns:
            Tuple of (modified_df, error_message)
            - On success: (df, None)
            - On failure: (None, error_string)
        """
        # Create a copy to avoid modifying the original
        df_copy = df.copy()

        # Create local namespace with the dataframe
        local_vars = {'df': df_copy}

        try:
            # Execute the code
            exec(code, self.safe_globals, local_vars)

            # Get the modified dataframe
            result_df = local_vars['df']

            # Validate the output
            validation_error = self._validate_output(df, result_df)
            if validation_error:
                return None, validation_error

            return result_df, None

        except SyntaxError as e:
            return None, f"Syntax Error: {str(e)}"

        except KeyError as e:
            return None, f"Column not found: {str(e)}"

        except Exception as e:
            # Capture full traceback for debugging
            tb = traceback.format_exc()
            return None, f"Execution Error: {str(e)}\n{tb}"

    def _validate_output(
        self,
        original_df: pd.DataFrame,
        new_df: pd.DataFrame
    ) -> Optional[str]:
        """
        Validate the output dataframe

        Checks:
        - Row count matches original
        - No complete NaN columns added
        - DataFrame is returned

        Returns:
            Error message if invalid, None if valid
        """
        if new_df is None:
            return "Code did not return a valid dataframe"

        if not isinstance(new_df, pd.DataFrame):
            return f"Expected DataFrame, got {type(new_df)}"

        # Check row count
        if len(new_df) != len(original_df):
            return (
                f"Row count mismatch: original={len(original_df)}, "
                f"new={len(new_df)}"
            )

        # Check for new columns that are completely NaN
        original_cols = set(original_df.columns)
        new_cols = set(new_df.columns) - original_cols

        for col in new_cols:
            if new_df[col].isna().all():
                return f"New column '{col}' is completely NaN"

        return None

    def get_new_columns(
        self,
        original_df: pd.DataFrame,
        new_df: pd.DataFrame
    ) -> list[str]:
        """Get list of newly created columns"""
        original_cols = set(original_df.columns)
        new_cols = set(new_df.columns)
        return list(new_cols - original_cols)
