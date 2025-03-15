import dataset
from KNN import KNN
import os
import numpy as np
def accuracy_function(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total
def recall_function(y_true, y_pred):
    # 真陽/(真陽+假陰)
    true_positives = np.sum((y_true == 1) & (y_pred == 1))  # 預測為1，實際為1
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))  # 預測為0，實際為1
    return true_positives / (true_positives + false_negatives)
def process_sigle_dataset(train_path, test_path, missing_handle = True):
    Train_DataSet = dataset.Dataset(train_path)
    Train_DataSet.Read_data()
    Train_DataSet.Process_data(missing_handle)
    x_train, y_train = Train_DataSet.get_Process_data()

    Test_DataSet = dataset.Dataset(test_path)
    Test_DataSet.Read_data()
    Test_DataSet.Process_data(missing_handle)
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
    accuracy = accuracy_function(y_test, y_prediction)
    recall = recall_function(y_test, y_prediction)
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