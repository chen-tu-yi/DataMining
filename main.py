import dataset
from KNN import KNN
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import os
# scikit-learn套件僅用於計算準確率和召回率
def process_sigle_dataset(train_path, test_path):
    Train_DataSet = dataset.Dataset(train_path)
    Train_DataSet.Read_data()
    Train_DataSet.Process_data()
    x_train, y_train = Train_DataSet.get_Process_data()

    Test_DataSet = dataset.Dataset(test_path)
    Test_DataSet.Read_data()
    Test_DataSet.Process_data()
    x_test, y_test = Test_DataSet.get_Process_data()

    # 交叉驗證
    k_value = range(1,21,2)
    knn_model = KNN(5) # 預設
    best_k = knn_model.cross_validation_for_k(x_train, y_train, k_value, cv=5)
    print(f"最佳 K 值: {best_k}")
    knn_model.k = best_k
    knn_model.fit(x_train, y_train)
    # 預測及評估
    y_prediction = knn_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_prediction)
    recall = recall_score(y_test, y_prediction)
    #print(f"KNN 測試集準確率: {accuracy:.4f}")
    print("-" * 50)
    return accuracy, recall

def main():
    print("\n")
    dataset_paths=[
        (os.path.join("test_A", "train_data.csv"),os.path.join("test_A", "test_data.csv")),
        (os.path.join("test_B", "train_data.csv"),os.path.join("test_B", "test_data.csv"))
    ]
    accuraciesAndrecalls = []
    for train_path,test_path in dataset_paths:
        accuracy, recall = process_sigle_dataset(train_path, test_path)
        accuraciesAndrecalls.append((train_path, accuracy, recall))
    print("\n=== 所有數據集的準確率 ===")
    for train_path, acc, recall in accuraciesAndrecalls:
        print(f"{train_path[:6]}: 準確率 = {acc:.4f}")
        print(f"{train_path[:6]}: 召回率 = {recall:.4f}")
if __name__ == '__main__':
    main()