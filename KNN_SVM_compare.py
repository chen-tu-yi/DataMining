import os
import numpy as np
import pandas as pd
import main
import dataset
from KNN import KNN
from SVM import SVM
from main import accuracy_function
from main import recall_function

def process_single_dataset(train_path, test_path, missing_handle=True):
    Train_DataSet = dataset.Dataset(train_path)
    Train_DataSet.Read_data()
    Train_DataSet.Process_data(missing_handle)
    x_train, y_train = Train_DataSet.get_Process_data()

    Test_DataSet = dataset.Dataset(test_path)
    Test_DataSet.Read_data()
    Test_DataSet.Process_data(missing_handle)
    x_test, y_test = Test_DataSet.get_Process_data()

    k_values = range(1,21,2)
    knn_model = KNN(5)
    best_k = knn_model.cross_validation_for_k(x_train, y_train, k_values, cv=5)
    knn_model.k = best_k
    knn_model.fit(x_train, y_train)
    y_pred_knn = knn_model.predict(x_test)
    knn_acc = accuracy_function(y_test, y_pred_knn)
    knn_recall = recall_function(y_test, y_pred_knn)

    svm_model = SVM(kernel='rbf', C=1)
    svm_model.fit(x_train, y_train)
    y_pred_svm = svm_model.predict(x_test)
    svm_acc = accuracy_function(y_test, y_pred_svm)
    svm_recall = recall_function(y_test, y_pred_svm)

    return knn_acc, knn_recall, svm_acc, svm_recall

def main():
    dataset_paths = [
        (os.path.join("test_A", "train_data.csv"), os.path.join("test_A", "test_data.csv")),
        (os.path.join("test_B", "train_data.csv"), os.path.join("test_B", "test_data.csv"))
    ]

    results = []
    for train_path, test_path in dataset_paths:
        knn_acc, knn_recall, svm_acc, svm_recall = process_single_dataset(train_path, test_path)
        dataset_label = train_path[:6]
        results.append({
            "Dataset": dataset_label,
            "KNN_Accuracy": knn_acc,
            "KNN_Recall": knn_recall,
            "SVM_Accuracy": svm_acc,
            "SVM_Recall": svm_recall
        })

    print("\n=== 比較結果 ===")
    for r in results:
        print(f"{r['Dataset']}: KNN 準確率 = {r['KNN_Accuracy']:.4f}，召回率 = {r['KNN_Recall']:.4f}")
        print(f"{r['Dataset']}: SVM 準確率 = {r['SVM_Accuracy']:.4f}，召回率 = {r['SVM_Recall']:.4f}")
        print("-" * 50)

    # export
    df_results = pd.DataFrame(results)
    output_csv = "KNN_SVM_Comparison.csv"
    df_results.to_csv(output_csv, index=False)

if __name__ == '__main__':
    main()