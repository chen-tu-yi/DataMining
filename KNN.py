import numpy as np

class KNN:
    def __init__(self, k=5, distance_method='manhattan'):
        self.k = k
        self.distance_method = distance_method
        self.x_train = None
        self.y_train = None
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    def compute_distance(self, point_1, point_2):
        if(self.distance_method == 'euclidean'):
            return np.sqrt(np.sum((point_1 - point_2) ** 2))
        elif(self.distance_method == 'manhattan'):
            return np.sum(np.abs(point_1 - point_2))
    def predict(self, x_test): # 對每個點進行預測
        predictions = []
        for point in x_test:
            distances = [self.compute_distance(point, train_point) for train_point in self.x_train]
            k_idx = np.argsort(distances)[:self.k] # 找出前 k 個最近的鄰居
            k_nearest_labels = [self.y_train[i] for i in k_idx] # 鄰居的分類(Outcome)
            most = np.argmax(np.bincount(k_nearest_labels)) # 投票結果
            predictions.append(most)
        return np.array(predictions)
    def evaluate(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return np.mean(y_predict == y_test)
    def cross_validation(self, x, y,folds = 5):
        fold_sise = len(x) // folds
        scores = []
        for i in range(folds):
            test_start = i*fold_sise 
            test_end = (i+1)*fold_sise if i!=folds else len(x)
            x_train = np.concatenate([x[:test_start],x[test_end:]], axis = 0)
            y_train = np.concatenate([y[:test_start],y[test_end:]], axis = 0)
            x_test = x[test_start:test_end]
            y_test = y[test_start:test_end]
            self.fit(x_train, y_train)
            scores.append(self.evaluate(x_test, y_test))
        return np.mean(scores)
    def cross_validation_for_k(self, x, y, k_value, cv=5):
        best_k = None
        best_score = 0
        for k in k_value:
            self.k = k
            score = self.cross_validation(x, y, cv)
            if(score > best_score):
                best_score = score
                best_k = k
        return best_k


        