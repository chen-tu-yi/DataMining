import os
from main import process_sigle_dataset
import pandas as pd
def main():
    dataset_paths = [
        (os.path.join("test_A", "train_data.csv"), os.path.join("test_A", "test_data.csv")),
        (os.path.join("test_B", "train_data.csv"), os.path.join("test_B", "test_data.csv"))
    ]
    results = []
    for train_path, test_path in dataset_paths:
        accuracy_with_missing_handle, recall_with_missing_handle = process_sigle_dataset(train_path, test_path, missing_handle=True)
        accuracy_without_missing_handle, recall_without_missing_handle = process_sigle_dataset(train_path, test_path, missing_handle=False)
        results.append({
            "Dataset": train_path[:6],
            "With Missing Handle Accuracy": accuracy_with_missing_handle,
            "Without Missing Handle Accuracy": accuracy_without_missing_handle,
            "With Missing Handle Recall": recall_with_missing_handle,
            "Without Missing Handle Recall": recall_without_missing_handle
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv("accuracy_recall_comparison.csv", index=False)
    print("\n=== 準確率和召回率比較 ===")
    print(df_results)

if __name__ == "__main__":
    main()