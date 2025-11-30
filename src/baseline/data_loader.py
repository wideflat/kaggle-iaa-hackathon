"""
Data loading utilities for Ames Housing dataset
"""

import pandas as pd
import os


# Column name mapping from AmesHousing.csv to Kaggle format
AMES_TO_KAGGLE_COLUMNS = {
    'Order': 'Id',
    'MS SubClass': 'MSSubClass',
    'MS Zoning': 'MSZoning',
    'Lot Frontage': 'LotFrontage',
    'Lot Area': 'LotArea',
    'Lot Shape': 'LotShape',
    'Land Contour': 'LandContour',
    'Lot Config': 'LotConfig',
    'Land Slope': 'LandSlope',
    'Condition 1': 'Condition1',
    'Condition 2': 'Condition2',
    'Bldg Type': 'BldgType',
    'House Style': 'HouseStyle',
    'Overall Qual': 'OverallQual',
    'Overall Cond': 'OverallCond',
    'Year Built': 'YearBuilt',
    'Year Remod/Add': 'YearRemodAdd',
    'Roof Style': 'RoofStyle',
    'Roof Matl': 'RoofMatl',
    'Exterior 1st': 'Exterior1st',
    'Exterior 2nd': 'Exterior2nd',
    'Mas Vnr Type': 'MasVnrType',
    'Mas Vnr Area': 'MasVnrArea',
    'Exter Qual': 'ExterQual',
    'Exter Cond': 'ExterCond',
    'Bsmt Qual': 'BsmtQual',
    'Bsmt Cond': 'BsmtCond',
    'Bsmt Exposure': 'BsmtExposure',
    'BsmtFin Type 1': 'BsmtFinType1',
    'BsmtFin SF 1': 'BsmtFinSF1',
    'BsmtFin Type 2': 'BsmtFinType2',
    'BsmtFin SF 2': 'BsmtFinSF2',
    'Bsmt Unf SF': 'BsmtUnfSF',
    'Total Bsmt SF': 'TotalBsmtSF',
    'Heating QC': 'HeatingQC',
    'Central Air': 'CentralAir',
    '1st Flr SF': '1stFlrSF',
    '2nd Flr SF': '2ndFlrSF',
    'Low Qual Fin SF': 'LowQualFinSF',
    'Gr Liv Area': 'GrLivArea',
    'Bsmt Full Bath': 'BsmtFullBath',
    'Bsmt Half Bath': 'BsmtHalfBath',
    'Full Bath': 'FullBath',
    'Half Bath': 'HalfBath',
    'Bedroom AbvGr': 'Bedroom',
    'Kitchen AbvGr': 'Kitchen',
    'Kitchen Qual': 'KitchenQual',
    'TotRms AbvGrd': 'TotRmsAbvGrd',
    'Fireplace Qu': 'FireplaceQu',
    'Garage Type': 'GarageType',
    'Garage Yr Blt': 'GarageYrBlt',
    'Garage Finish': 'GarageFinish',
    'Garage Cars': 'GarageCars',
    'Garage Area': 'GarageArea',
    'Garage Qual': 'GarageQual',
    'Garage Cond': 'GarageCond',
    'Paved Drive': 'PavedDrive',
    'Wood Deck SF': 'WoodDeckSF',
    'Open Porch SF': 'OpenPorchSF',
    'Enclosed Porch': 'EnclosedPorch',
    '3Ssn Porch': '3SsnPorch',
    'Screen Porch': 'ScreenPorch',
    'Pool Area': 'PoolArea',
    'Pool QC': 'PoolQC',
    'Misc Feature': 'MiscFeature',
    'Misc Val': 'MiscVal',
    'Mo Sold': 'MoSold',
    'Yr Sold': 'YrSold',
    'Sale Type': 'SaleType',
    'Sale Condition': 'SaleCondition',
}


class AmesDataLoader:
    """Load and provide access to Ames Housing data"""

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.train_df = None
        self.test_df = None
        self.data_description = None

    def load_ames_housing(self) -> pd.DataFrame:
        """
        Load AmesHousing.csv and convert to Kaggle column format.

        Returns:
            DataFrame with Kaggle-style column names
        """
        ames_path = os.path.join(self.data_dir, 'AmesHousing.csv')
        df = pd.read_csv(ames_path)

        # Drop PID column (not in Kaggle data)
        df = df.drop('PID', axis=1)

        # Rename columns to Kaggle format (spaces removed, etc.)
        df = df.rename(columns=AMES_TO_KAGGLE_COLUMNS)

        # Reassign Id to avoid conflicts (start after max Kaggle Id)
        df['Id'] = df['Id'] + 10000

        return df

    def load_train(self, include_ames: bool = False) -> pd.DataFrame:
        """
        Load training data.

        Args:
            include_ames: If True, include AmesHousing.csv data

        Returns:
            Training DataFrame
        """
        train_path = os.path.join(self.data_dir, 'train.csv')
        self.train_df = pd.read_csv(train_path)

        if include_ames:
            ames_df = self.load_ames_housing()
            self.train_df = pd.concat([self.train_df, ames_df], ignore_index=True)
            print(f"   Added AmesHousing.csv: {len(ames_df)} rows")
            print(f"   Total training rows: {len(self.train_df)}")

        return self.train_df

    def load_test(self):
        """Load test data"""
        test_path = os.path.join(self.data_dir, 'test.csv')
        self.test_df = pd.read_csv(test_path)
        return self.test_df

    def load_description(self):
        """Load data description file"""
        desc_path = os.path.join(self.data_dir, 'data_description.txt')
        with open(desc_path, 'r') as f:
            self.data_description = f.read()
        return self.data_description

    def get_train_data(self):
        """Get training features and target"""
        if self.train_df is None:
            self.load_train()

        X = self.train_df.drop(['Id', 'SalePrice'], axis=1)
        y = self.train_df['SalePrice']
        ids = self.train_df['Id']

        return X, y, ids

    def get_test_data(self):
        """Get test features"""
        if self.test_df is None:
            self.load_test()

        X = self.test_df.drop(['Id'], axis=1)
        ids = self.test_df['Id']

        return X, ids

    def get_all_data(self):
        """Load all data"""
        train = self.load_train()
        test = self.load_test()
        desc = self.load_description()

        return {
            'train': train,
            'test': test,
            'description': desc
        }
