from scipy import sparse
import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticRegressionDP(object):
    def __init__(self, ):
        self.model = LogisticRegression(C=1, random_state=np.random.RandomState(42), max_iter=1000)
        self.trained=False
        
    def predict(self, collection, item_representation):
        assert self.trained        
        if type(item_representation[list(item_representation)[0]])==np.ndarray:
            X = np.vstack([item_representation[item.id_] for item in collection])
        else:
            X = sparse.vstack([item_representation[item.id_] for item in collection])
        return self.model.predict_proba(X)[:,1]
    
    def fit(self, collection, item_representation):
#         X = sparse.vstack([item_representation[item.id_] for item in collection])
        if type(item_representation[list(item_representation)[0]])==np.ndarray:
            X = np.vstack([item_representation[item.id_] for item in collection])
        else:
            X = sparse.vstack([item_representation[item.id_] for item in collection])
        y = np.hstack([1 if item.is_relevant() else 0 for item in collection])
        self.model.fit(X, y)
        self.trained=True
        
from sklearn.svm import SVC 
class SVMDP(object):
    def __init__(self, ):
        self.model = SVC(C=1, random_state=np.random.RandomState(42), probability=True)
        self.trained=False
        
    def predict(self, collection, item_representation):
        assert self.trained
#         X = sparse.vstack([item_representation[item.id_] for item in collection])

        if type(item_representation[list(item_representation)[0]])==np.ndarray:
            X = np.vstack([item_representation[item.id_] for item in collection])
        else:
            X = sparse.vstack([item_representation[item.id_] for item in collection])
        return self.model.predict_proba(X)[:,1]
    
    def fit(self, collection, item_representation):
#         X = sparse.vstack([item_representation[item.id_] for item in collection])
        if type(item_representation[list(item_representation)[0]])==np.ndarray:
            X = np.vstack([item_representation[item.id_] for item in collection])
        else:
            X = sparse.vstack([item_representation[item.id_] for item in collection])
        y = np.hstack([1 if item.is_relevant() else 0 for item in collection])
        self.model.fit(X, y)
        self.trained=True
    