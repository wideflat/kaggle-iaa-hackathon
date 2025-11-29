"""
Data loading utilities for Ames Housing dataset
"""

import pandas as pd
import os


class AmesDataLoader:
    """Load and provide access to Ames Housing data"""

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.train_df = None
        self.test_df = None
        self.data_description = None

    def load_train(self):
        """Load training data"""
        train_path = os.path.join(self.data_dir, 'train.csv')
        self.train_df = pd.read_csv(train_path)
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
