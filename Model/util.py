import pandas as pd
from scipy.io.arff import loadarff


class CSVStream:
    def __init__(self, csv_file: str, target: str = None) -> None:
        self.csv_file = csv_file
        self.data = pd.read_csv(self.csv_file)
        if target is None:
            self.target = self.data.columns[-1]
        self.classes = self.data[self.target].unique()
        self.n_classes = len(self.classes)
        self.n_features = self.data.shape[1] - 1
        self.n_samples = self.data.shape[0]
        self.index = 0

    def __iter__(self):
        while True:
            row = self.data.iloc[self.index, :-1]
            x = row.to_dict()
            y = self.data.iloc[self.index, -1]
            self.index += 1
            if self.index >= self.n_samples:
                break
            yield x, y


class ARFFStream:
    def __init__(self, arff_file: str, target: str = None) -> None:
        self.arff_file = arff_file
        arff_loaded = loadarff(self.arff_file)
        self.data = pd.DataFrame(arff_loaded[0])

        if target is None:
            self.target = self.data.columns[-1]
        codes, uniques = pd.factorize(self.data.iloc[:, -1])
        self.data.iloc[:, -1] = codes
        self.classes = self.data[self.target].unique()
        if cat_cols := [col for col in self.data.columns if self.data[col].dtype == "O"]:
            for col in cat_cols:
                codes, uniques = pd.factorize(self.data[col])
                self.data[col] = codes

        self.n_classes = len(self.classes)
        self.n_features = self.data.shape[1] - 1
        self.n_samples = self.data.shape[0]
        self.features = self.data.columns[:-1].tolist()
        self.index = 0

    def __iter__(self):
        while True:
            if self.index >= self.n_samples:
                break
            row = self.data.iloc[self.index, :-1]
            x = row.to_dict()
            y = self.data.iloc[self.index, -1]
            self.index += 1
            yield x, y
