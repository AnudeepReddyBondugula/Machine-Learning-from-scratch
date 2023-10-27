import numpy as np


class KNN_Classifier:
    def __init__(self, K=5):
        pass

    def fit(self, X, y):
        pass
        

    def eucledian_distance(self, record1, record2):
        diff = record1 - record2 # [(x1-x2), (y1-y2), (z1-z2)]
        eucledian_dist = np.sqrt(np.sum(diff * diff)) # np.linalg.norm(diff)

        return eucledian_dist 
    
    