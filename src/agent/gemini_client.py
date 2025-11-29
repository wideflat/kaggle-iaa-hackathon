"""
Gemini API client for feature engineering code generation
"""

import os
import re
from dotenv import load_dotenv
import google.generativeai as genai


class GeminiClient:
    """
    Wrapper for Google Gemini API to generate feature engineering code
    """

    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        """Initialize Gemini client with API key from .env"""
        load_dotenv()

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Create a .env file with GEMINI_API_KEY=your_key"
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_feature(
        self,
        data_description: str,
        column_info: str,
        existing_features: list[str] | None = None
    ) -> str:
        """
        Generate a new feature engineering code snippet

        Args:
            data_description: Content of data_description.txt
            column_info: Comma-separated list of column names
            existing_features: List of already tried feature codes to avoid

        Returns:
            Python code string for creating a new feature
        """
        prompt = self._build_prompt(data_description, column_info, existing_features)
        response = self.model.generate_content(prompt)
        return self._extract_code(response.text)

    def _build_prompt(
        self,
        data_description: str,
        column_info: str,
        existing_features: list[str] | None = None
    ) -> str:
        """Build the prompt for feature generation"""

        # Truncate data description if too long
        max_desc_len = 3000
        if len(data_description) > max_desc_len:
            data_description = data_description[:max_desc_len] + "\n... (truncated)"

        prompt = f"""You are an expert data scientist working on the Ames Housing dataset for house price prediction.

## Data Description (partial):
{data_description}

## Available Columns:
{column_info}

## Task:
Generate ONE new feature that will help predict house sale prices (SalePrice).
The feature should be meaningful for real estate valuation.

## Requirements:
1. Return ONLY executable Python code
2. Assume the dataframe variable is named 'df'
3. Handle potential NaN values with .fillna()
4. The feature should have a clear semantic meaning
5. Name the feature descriptively

## Example format:
```python
# Feature: Quality-Size Interaction
df['QualSF'] = df['OverallQual'] * df['GrLivArea']
```

## Good feature ideas for housing:
- Interactions between quality and size
- Age-related calculations (year differences)
- Area aggregations (total outdoor space, total indoor space)
- Ratios (e.g., basement finished ratio)
- Binary indicators (has pool, has garage, etc.)

## Features that ALREADY EXIST (do NOT recreate these):
- TotalSF (TotalBsmtSF + 1stFlrSF + 2ndFlrSF)
- HouseAge (YrSold - YearBuilt)
- RemodAge (YrSold - YearRemodAdd)
- TotalBath (FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath)
- PorchArea (OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch)
"""

        if existing_features:
            prompt += f"""
## Already tried features (DO NOT repeat these):
{chr(10).join(existing_features[-10:])}
"""

        prompt += "\n## Generate ONE new feature:\n"
        return prompt

    def _extract_code(self, response: str) -> str:
        """
        Extract Python code from LLM response

        Handles markdown code blocks and plain text responses
        """
        # Try to extract from markdown code block
        code_block_pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            # Return the first code block found
            code = matches[0].strip()
        else:
            # No code block, assume the response is raw code
            # Remove any leading/trailing explanation text
            lines = response.strip().split('\n')
            code_lines = []
            in_code = False

            for line in lines:
                # Start capturing when we see df[ or # Feature
                if 'df[' in line or line.strip().startswith('# Feature'):
                    in_code = True
                if in_code:
                    code_lines.append(line)

            code = '\n'.join(code_lines) if code_lines else response.strip()

        return code
