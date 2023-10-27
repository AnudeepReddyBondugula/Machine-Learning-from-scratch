import numpy as np
import pandas as pd
from collections import Counter



class KNN_Classifier:
    def __init__(self, K=5):
        self.K = K

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        self.predictions = []
        for index, test_record in X_test.iterrows():
            distances = []
            for index, train_record in self.X_train.iterrows():
                distances.append(self.eucledian_distance(test_record, train_record))
            distances = sorted(range(len(distances)), key=lambda x: distances[x])[:self.K]

            distances = [self.y_train.iloc[x] for x in distances]
            self.predictions.append(self.get_highest_frequency(distances))
        return self.predictions
    
    def get_highest_frequency(self, strings):
        string_counts = Counter(strings)
        most_common_strings = [string for string, count in string_counts.items() if count == max(string_counts.values())]

        return most_common_strings[0]

    def eucledian_distance(self, record1, record2):

        diff = record1 - record2 # [(x1-x2), (y1-y2), (z1-z2)]
        # print(diff)
        eucledian_dist = np.sqrt(np.sum(diff * diff)) # np.linalg.norm(diff)

        return eucledian_dist 

    def accuracy_score(self, y_test):
        number_of_records = len(self.predictions)
        correct = 0
        for i in range(number_of_records):
            if self.predictions[i] == y_test.iloc[i]:
                correct += 1
        return correct/number_of_records
    
    