"""
Gemini API client for feature engineering code generation

Enhanced with:
- Multiple few-shot examples
- Feature strategy selection
- Better duplicate avoidance
"""

import os
import re
import random
from dotenv import load_dotenv
import google.generativeai as genai


# Few-shot examples organized by strategy
FEW_SHOT_EXAMPLES = {
    'foundational': [
        {
            'name': 'TotalSF',
            'code': "df['TotalSF'] = df['TotalBsmtSF'].fillna(0) + df['1stFlrSF'] + df['2ndFlrSF'].fillna(0)",
            'rationale': 'Total square footage - the most important size metric for house value'
        },
        {
            'name': 'HouseAge',
            'code': "df['HouseAge'] = df['YrSold'] - df['YearBuilt']",
            'rationale': 'Age of house at time of sale - older houses typically worth less'
        },
        {
            'name': 'TotalBath',
            'code': "df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'].fillna(0) + 0.5*df['BsmtHalfBath'].fillna(0)",
            'rationale': 'Total bathroom count (half baths weighted 0.5) - key amenity metric'
        },
        {
            'name': 'TotalPorchSF',
            'code': "df['TotalPorchSF'] = df['OpenPorchSF'].fillna(0) + df['EnclosedPorch'].fillna(0) + df['3SsnPorch'].fillna(0) + df['ScreenPorch'].fillna(0)",
            'rationale': 'Total porch area - outdoor living space adds value'
        },
        {
            'name': 'RemodAge',
            'code': "df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']",
            'rationale': 'Years since remodel - recently remodeled homes worth more'
        },
    ],
    'quality_interactions': [
        {
            'name': 'OverallQual_ExterQual',
            'code': "df['OverallQual_ExterQual'] = df['OverallQual'] * df['ExterQual']",
            'rationale': 'Interaction between overall and exterior quality - both matter for curb appeal'
        },
        {
            'name': 'KitchenQual_GrLivArea',
            'code': "df['KitchenQual_GrLivArea'] = df['KitchenQual'] * df['GrLivArea']",
            'rationale': 'Kitchen quality weighted by living area - good kitchens in big houses worth more'
        },
        {
            'name': 'TotalQualScore',
            'code': "df['TotalQualScore'] = df['OverallQual'] + df['ExterQual'] + df['KitchenQual'] + df['BsmtQual']",
            'rationale': 'Combined quality score across all quality dimensions'
        },
    ],
    'quality_tiers': [
        {
            'name': 'QualityTier',
            'code': "df['QualityTier'] = pd.cut(df['OverallQual'], bins=[0, 4, 6, 8, 10], labels=[1, 2, 3, 4]).astype(int)",
            'rationale': 'Categorize quality into tiers - captures non-linear quality bands'
        },
        {
            'name': 'IsHighEnd',
            'code': "df['IsHighEnd'] = (df['OverallQual'] >= 8).astype(int)",
            'rationale': 'Binary indicator for high-end homes (quality 8+) - luxury premium'
        },
        {
            'name': 'QualCondProduct',
            'code': "df['QualCondProduct'] = df['OverallQual'] * df['OverallCond']",
            'rationale': 'Quality-condition product - both matter for value'
        },
    ],
    'size_efficiency': [
        {
            'name': 'SFPerBedroom',
            'code': "df['SFPerBedroom'] = df['GrLivArea'] / (df['BedroomAbvGr'].replace(0, 1))",
            'rationale': 'Square feet per bedroom - larger bedrooms indicate luxury'
        },
        {
            'name': 'BasementRatio',
            'code': "df['BasementRatio'] = df['TotalBsmtSF'].fillna(0) / (df['GrLivArea'].replace(0, 1))",
            'rationale': 'Basement to living area ratio - more basement = more value'
        },
        {
            'name': 'FinishedRatio',
            'code': "df['FinishedRatio'] = (df['BsmtFinSF1'].fillna(0) + df['BsmtFinSF2'].fillna(0)) / (df['TotalBsmtSF'].fillna(1).replace(0, 1))",
            'rationale': 'Finished basement percentage - finished space worth more'
        },
    ],
    'interaction': [
        {
            'name': 'QualitySF',
            'code': "df['QualitySF'] = df['OverallQual'] * df['GrLivArea']",
            'rationale': 'Quality multiplied by size - high quality large homes are worth more than sum of parts'
        },
        {
            'name': 'GarageScore',
            'code': "df['GarageScore'] = df['GarageArea'].fillna(0) * df['GarageCars'].fillna(0)",
            'rationale': 'Garage capacity interaction - larger garages with more car capacity add value'
        },
        {
            'name': 'BsmtScore',
            'code': "df['BsmtScore'] = df['TotalBsmtSF'].fillna(0) * df['BsmtQual'].fillna(0)",
            'rationale': 'Basement quality-size interaction - finished basement quality matters'
        },
    ],
    'ratio': [
        {
            'name': 'BsmtFinRatio',
            'code': "df['BsmtFinRatio'] = df['BsmtFinSF1'].fillna(0) / (df['TotalBsmtSF'].fillna(1).replace(0, 1))",
            'rationale': 'Ratio of finished basement - more finished = more usable space'
        },
        {
            'name': 'LotDepth',
            'code': "df['LotDepth'] = df['LotArea'] / (df['LotFrontage'].fillna(1).replace(0, 1))",
            'rationale': 'Lot depth approximation - deep lots may have different value than wide lots'
        },
        {
            'name': 'LivAreaPerRoom',
            'code': "df['LivAreaPerRoom'] = df['GrLivArea'] / (df['TotRmsAbvGrd'].replace(0, 1))",
            'rationale': 'Average room size - larger rooms indicate luxury'
        },
    ],
    'aggregation': [
        {
            'name': 'TotalOutdoorSF',
            'code': "df['TotalOutdoorSF'] = df['WoodDeckSF'].fillna(0) + df['OpenPorchSF'].fillna(0) + df['EnclosedPorch'].fillna(0) + df['PoolArea'].fillna(0)",
            'rationale': 'Total outdoor living space - outdoor amenities add value'
        },
        {
            'name': 'TotalQual',
            'code': "df['TotalQual'] = df['OverallQual'] + df['OverallCond']",
            'rationale': 'Combined quality and condition score'
        },
        {
            'name': 'TotalFinishedSF',
            'code': "df['TotalFinishedSF'] = df['BsmtFinSF1'].fillna(0) + df['BsmtFinSF2'].fillna(0) + df['1stFlrSF'] + df['2ndFlrSF'].fillna(0)",
            'rationale': 'Total finished living space including basement'
        },
    ],
    'binary': [
        {
            'name': 'HasPool',
            'code': "df['HasPool'] = (df['PoolArea'].fillna(0) > 0).astype(int)",
            'rationale': 'Binary indicator for pool presence - pools can add significant value'
        },
        {
            'name': 'HasGarage',
            'code': "df['HasGarage'] = (df['GarageArea'].fillna(0) > 0).astype(int)",
            'rationale': 'Binary indicator for garage presence'
        },
        {
            'name': 'IsNew',
            'code': "df['IsNew'] = (df['YearBuilt'] == df['YrSold']).astype(int)",
            'rationale': 'Binary indicator for new construction - new homes have premium'
        },
        {
            'name': 'HasFireplace',
            'code': "df['HasFireplace'] = (df['Fireplaces'].fillna(0) > 0).astype(int)",
            'rationale': 'Binary indicator for fireplace presence'
        },
    ],
    'polynomial': [
        {
            'name': 'QualSquared',
            'code': "df['QualSquared'] = df['OverallQual'] ** 2",
            'rationale': 'Squared quality - captures non-linear quality premium at high end'
        },
        {
            'name': 'LogLotArea',
            'code': "df['LogLotArea'] = np.log1p(df['LotArea'])",
            'rationale': 'Log-transformed lot area - diminishing returns for very large lots'
        },
        {
            'name': 'GrLivAreaSq',
            'code': "df['GrLivAreaSq'] = df['GrLivArea'] ** 2",
            'rationale': 'Squared living area - non-linear size premium'
        },
    ],
    'temporal': [
        {
            'name': 'YearsSinceRemod',
            'code': "df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']",
            'rationale': 'Years since last remodel - recently remodeled homes worth more'
        },
        {
            'name': 'GarageAge',
            'code': "df['GarageAge'] = df['YrSold'] - df['GarageYrBlt'].fillna(df['YearBuilt'])",
            'rationale': 'Age of garage - newer garages may add value'
        },
        {
            'name': 'WasRemodeled',
            'code': "df['WasRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)",
            'rationale': 'Binary indicator if house was ever remodeled'
        },
    ],
}

# Features that already exist in baseline preprocessor
# NOTE: With ultra-minimal preprocessing, NONE of these exist yet!
# The agent should CREATE these foundational features.
BASELINE_FEATURES = [
    # Empty - agent discovers everything in ultra-minimal mode
]


class GeminiClient:
    """
    Wrapper for Google Gemini API to generate feature engineering code

    Features:
    - Multiple few-shot examples organized by strategy
    - Strategy-specific prompting
    - Better duplicate avoidance
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
        self.strategies = list(FEW_SHOT_EXAMPLES.keys())

    def generate_feature(
        self,
        data_description: str,
        column_info: str,
        existing_features: list[str] | None = None,
        strategy: str | None = None
    ) -> str:
        """
        Generate a new feature engineering code snippet

        Args:
            data_description: Content of data_description.txt
            column_info: Comma-separated list of column names
            existing_features: List of already tried feature codes to avoid
            strategy: Optional strategy hint ('interaction', 'ratio', 'aggregation',
                     'binary', 'polynomial', 'temporal'). If None, randomly selected.

        Returns:
            Python code string for creating a new feature
        """
        prompt = self._build_prompt(
            data_description,
            column_info,
            existing_features,
            strategy
        )
        response = self.model.generate_content(prompt)
        return self._extract_code(response.text)

    def _build_prompt(
        self,
        data_description: str,
        column_info: str,
        existing_features: list[str] | None = None,
        strategy: str | None = None
    ) -> str:
        """Build the prompt for feature generation with few-shot examples"""

        # Truncate data description if too long
        max_desc_len = 2500
        if len(data_description) > max_desc_len:
            data_description = data_description[:max_desc_len] + "\n... (truncated)"

        # Select strategy (random if not specified)
        if strategy is None:
            strategy = random.choice(self.strategies)

        # Get few-shot examples for this strategy
        examples = FEW_SHOT_EXAMPLES.get(strategy, [])
        example_text = self._format_examples(examples)

        prompt = f"""You are an expert data scientist specializing in real estate valuation and the Ames Housing dataset.

## Your Task
Generate ONE new feature to improve house price prediction. Focus on {strategy.upper()} features.

## Data Description
{data_description}

## Available Columns
{column_info}

## Few-Shot Examples ({strategy} features)
{example_text}

## Requirements
1. Return ONLY executable Python code
2. Use 'df' as the dataframe variable
3. Handle NaN values with .fillna() to avoid errors
4. Start with a comment: # Feature: <descriptive name>
5. Create a NOVEL feature not in the existing list below
6. DO NOT use import statements - 'np' (numpy) and 'pd' (pandas) are already available

## Features that ALREADY EXIST (do NOT recreate):
{chr(10).join('- ' + f for f in BASELINE_FEATURES)}
"""

        if existing_features:
            # Show recent attempts to avoid
            recent = existing_features[-15:]  # Last 15 attempts
            prompt += f"""
## Recently tried features (DO NOT repeat):
{chr(10).join(recent)}
"""

        prompt += f"""
## Strategy Hint: {strategy.upper()}
Think about {self._get_strategy_hint(strategy)}

## Generate ONE new {strategy} feature:
```python
"""
        return prompt

    def _format_examples(self, examples: list[dict]) -> str:
        """Format few-shot examples for the prompt"""
        formatted = []
        for ex in examples:
            formatted.append(
                f"Example - {ex['name']}:\n"
                f"```python\n# Feature: {ex['name']}\n{ex['code']}\n```\n"
                f"Rationale: {ex['rationale']}\n"
            )
        return '\n'.join(formatted)

    def _get_strategy_hint(self, strategy: str) -> str:
        """Get a hint for the given strategy"""
        hints = {
            'foundational': 'creating essential aggregate features like TotalSF, TotalBath, HouseAge - the basics that every good model needs',
            'quality_interactions': 'combining quality scores (OverallQual, ExterQual, KitchenQual are already 0-5 ordinal) with other features',
            'quality_tiers': 'binning quality scores into meaningful tiers (low/medium/high/luxury)',
            'size_efficiency': 'calculating ratios that measure space efficiency (SF per room, finished ratios)',
            'interaction': 'multiplying related features that together capture value (quality × size, etc.)',
            'ratio': 'dividing features to get proportions or per-unit metrics (finished ratio, per-room area)',
            'aggregation': 'summing related features into totals (outdoor space, total quality scores)',
            'binary': 'creating 0/1 indicators for presence/absence of features (has pool, is new, etc.)',
            'polynomial': 'squared terms or log transforms to capture non-linear relationships',
            'temporal': 'time-based calculations using year columns (age, years since remodel)',
        }
        return hints.get(strategy, 'creating meaningful features for house price prediction')

    def generate_feature_with_feedback(
        self,
        data_description: str,
        column_info: str,
        shap_summary: str,
        feature_insights: dict,
        existing_features: list[str] | None = None,
        strategy: str | None = None
    ) -> str:
        """
        Generate a new feature using SHAP-based feedback

        This is the reflection-enhanced version that uses feature importance
        insights to guide more targeted feature generation.

        Args:
            data_description: Content of data_description.txt
            column_info: Comma-separated list of column names
            shap_summary: Human-readable SHAP importance summary
            feature_insights: Dictionary with top_features, suggestions, etc.
            existing_features: List of already tried feature codes
            strategy: Optional strategy hint

        Returns:
            Python code string for creating a new feature
        """
        prompt = self._build_reflection_prompt(
            data_description,
            column_info,
            shap_summary,
            feature_insights,
            existing_features,
            strategy
        )
        response = self.model.generate_content(prompt)
        return self._extract_code(response.text)

    def _build_reflection_prompt(
        self,
        data_description: str,
        column_info: str,
        shap_summary: str,
        feature_insights: dict,
        existing_features: list[str] | None = None,
        strategy: str | None = None
    ) -> str:
        """Build a reflection prompt that uses SHAP insights"""

        # Truncate data description if too long
        max_desc_len = 2000
        if len(data_description) > max_desc_len:
            data_description = data_description[:max_desc_len] + "\n... (truncated)"

        # Select strategy based on insights if not specified
        if strategy is None:
            strategy = self._suggest_strategy_from_insights(feature_insights)

        # Get few-shot examples for this strategy
        examples = FEW_SHOT_EXAMPLES.get(strategy, [])
        example_text = self._format_examples(examples)

        # Format top features
        top_features = feature_insights.get('top_features', [])
        suggestions = feature_insights.get('suggestions', [])

        prompt = f"""You are an expert data scientist working on the Ames Housing dataset.
You have access to SHAP-based feature importance analysis from the current model.

## SHAP Analysis (What the model finds important)
{shap_summary}

## Key Insights
- Top predictive features: {', '.join(top_features[:5])}
- The model currently relies heavily on these features
- Consider creating features that INTERACT with or TRANSFORM these important features

## Suggestions Based on Analysis
{chr(10).join('- ' + s for s in suggestions[:4])}

## Data Description
{data_description}

## Available Columns
{column_info}

## Few-Shot Examples ({strategy} features)
{example_text}

## Requirements
1. Return ONLY executable Python code
2. Use 'df' as the dataframe variable
3. Handle NaN values with .fillna() to avoid errors
4. Start with a comment: # Feature: <descriptive name>
5. CREATE A FEATURE THAT LEVERAGES THE TOP IMPORTANT FEATURES
6. DO NOT use import statements - 'np' (numpy) and 'pd' (pandas) are already available
7. ONLY use columns that exist in the Available Columns list above

## Features that ALREADY EXIST (do NOT recreate):
{chr(10).join('- ' + f for f in BASELINE_FEATURES)}
"""

        if existing_features:
            recent = existing_features[-10:]
            prompt += f"""
## Recently tried features (DO NOT repeat):
{chr(10).join(recent)}
"""

        prompt += f"""
## Strategy: {strategy.upper()}
Focus on {self._get_strategy_hint(strategy)}

## IMPORTANT: Use insights from SHAP analysis!
The top feature is '{top_features[0] if top_features else 'unknown'}'. Consider:
- Interacting it with other features
- Creating ratio or polynomial transforms
- Combining related features

## Generate ONE new feature:
```python
"""
        return prompt

    def _suggest_strategy_from_insights(self, feature_insights: dict) -> str:
        """Suggest a strategy based on feature insights"""
        top_features = feature_insights.get('top_features', [])

        if not top_features:
            return random.choice(self.strategies)

        # Check what types of features are important
        top_str = ' '.join(top_features).lower()

        # Suggest strategy based on top features
        if any(x in top_str for x in ['qual', 'cond', 'overall']):
            return random.choice(['interaction', 'polynomial'])
        elif any(x in top_str for x in ['sf', 'area', 'sqft']):
            return random.choice(['ratio', 'aggregation'])
        elif any(x in top_str for x in ['year', 'yr', 'age']):
            return 'temporal'
        elif any(x in top_str for x in ['garage', 'bsmt', 'pool', 'fireplace']):
            return random.choice(['binary', 'interaction'])
        else:
            return random.choice(self.strategies)

    def generate_feature_batch(
        self,
        data_description: str,
        column_info: str,
        existing_features: list[str] | None = None,
        n_features: int = 5
    ) -> list[str]:
        """
        Generate multiple features in a single prompt.

        This is more efficient than generating one at a time and allows
        the model to create diverse features across different strategies.

        Args:
            data_description: Content of data_description.txt
            column_info: Comma-separated list of column names
            existing_features: List of already tried feature codes to avoid
            n_features: Number of features to generate (default: 5)

        Returns:
            List of Python code strings, each creating one feature
        """
        prompt = self._build_batch_prompt(
            data_description,
            column_info,
            existing_features,
            n_features
        )
        response = self.model.generate_content(prompt)
        return self._extract_multiple_codes(response.text)

    def _build_batch_prompt(
        self,
        data_description: str,
        column_info: str,
        existing_features: list[str] | None = None,
        n_features: int = 5
    ) -> str:
        """Build prompt for batch feature generation"""

        # Truncate data description if too long
        max_desc_len = 2000
        if len(data_description) > max_desc_len:
            data_description = data_description[:max_desc_len] + "\n... (truncated)"

        # Select diverse strategies
        strategies_to_use = random.sample(self.strategies, min(n_features, len(self.strategies)))

        # Get one example per strategy
        example_text = ""
        for strategy in strategies_to_use[:3]:
            examples = FEW_SHOT_EXAMPLES.get(strategy, [])
            if examples:
                ex = random.choice(examples)
                example_text += f"\n{strategy.upper()} Example:\n```python\n# Feature: {ex['name']}\n{ex['code']}\n```\n"

        prompt = f"""You are an expert data scientist specializing in the Ames Housing dataset.

## Your Task
Generate {n_features} DIFFERENT features to improve house price prediction.
Each feature should use a DIFFERENT strategy from: {', '.join(strategies_to_use)}

## Data Description
{data_description}

## Available Columns
{column_info}

## Example Features
{example_text}

## Requirements
1. Generate EXACTLY {n_features} features, each in a SEPARATE code block
2. Use 'df' as the dataframe variable
3. Handle NaN values with .fillna() to avoid errors
4. Start each feature with: # Feature: <descriptive name>
5. Each feature must be UNIQUE and use a different approach
6. DO NOT use import statements - 'np' (numpy) and 'pd' (pandas) are already available

## Features that ALREADY EXIST (do NOT recreate):
{chr(10).join('- ' + f for f in BASELINE_FEATURES)}
"""

        if existing_features:
            recent = existing_features[-10:]
            prompt += f"""
## Recently tried features (DO NOT repeat):
{chr(10).join(recent)}
"""

        prompt += f"""

## Generate {n_features} features (each in its own ```python block):
"""
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

    def _extract_multiple_codes(self, response: str) -> list[str]:
        """
        Extract multiple Python code blocks from LLM response.

        Args:
            response: Raw LLM response text

        Returns:
            List of code strings, one per feature
        """
        code_block_pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        codes = []
        for match in matches:
            code = match.strip()
            # Only include if it looks like feature code
            if 'df[' in code or '# Feature' in code:
                codes.append(code)

        return codes
