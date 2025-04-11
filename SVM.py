from sklearn.svm import SVC
import numpy as np
class SVM:
    def __init__(self, kernel='rbf', C=1):
        self.kernel = kernel
        self.C = C
        self.model = None

    def fit(self, x_train, y_train):
        self.model = SVC(kernel=self.kernel, C=self.C, random_state=40)
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.mean(y_pred == y_test) # accuracy

    def cross_validation(self, x, y, folds=5):
        fold_size = len(x) // folds
        scores = []
        for i in range(folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i != folds - 1 else len(x)
            x_train = np.concatenate([x[:test_start], x[test_end:]], axis=0)
            y_train = np.concatenate([y[:test_start], y[test_end:]], axis=0)
            x_test = x[test_start:test_end]
            y_test = y[test_start:test_end]
            self.fit(x_train, y_train)
            scores.append(self.evaluate(x_test, y_test))
        return np.mean(scores)

    def cross_validation_for_C(self, x, y, C_values, cv=5):
        best_C = None
        best_score = 0
        for c in C_values:
            self.C = c
            score = self.cross_validation(x, y, folds=cv)
            if score > best_score:
                best_score = score
                best_C = c
        return best_C