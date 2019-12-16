import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

class Data():
    def __init__(self, path2data, seed, drop_features=None):
        """
        Load data from CSV, keeps x, y, classes, class_weights and column_names
        path2data : str
            path to CSV file
        seed : int
            the seed used by the random number generator
        drop_features : list or None
            column labels to drop
        """
        self.seed = seed
        assert os.path.exists(path2data), '%s does not exist' % path2data
        df = pd.read_csv(path2data, sep=';')
        if drop_features is not None:
            df = df.drop(drop_features, axis=1)
        df = df.drop_duplicates()
        df = df.astype({'quality': 'float32'})

        self.x = df.loc[:, :'alcohol'].values
        self.y = df['quality'].values
        self.classes = np.unique(self.y).astype(np.int32)
        self.class_weights = {k: v for k, v in [(cl, len(self.y) * 1.0 / (len(self.classes) * np.sum(self.y == cl))) for cl in self.classes]}
        self.column_names = df.columns
        
    def scale(self):
        """
        Standardize a dataset along axis=0
        Center to the mean and component wise scale to unit variance.
        """
        self.x = preprocessing.scale(self.x)

    def get_train_test(self, test_size=0.3):
        """
        Split data on train and test set
        test_size : float, int
            the proportion of the dataset to include in the test split
        """
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
        train_index, test_index = next(sss.split(self.x, self.y))

        x_train = self.x[train_index, :]
        y_train = self.y[train_index]
        x_test = self.x[test_index, :]
        y_test = self.y[test_index]
        return x_train, y_train, x_test, y_test
