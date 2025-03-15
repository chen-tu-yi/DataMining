import pandas as pd
import os
class Dataset:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.x = None
        self.y = None
    def Read_data(self):
        self.data = pd.read_csv(self.path)
    def Process_data(self, missing_handle=True):
        self.x = self.data.drop(columns=["Outcome"])
        #self.x = self.data.drop(columns=["SkinThickness", "Pregnancies"]) #feature selection
        self.y = self.data["Outcome"].to_numpy()
        if missing_handle:
            for column in self.x.columns:
                mean_val = self.x[column][self.x[column] > 0].mean() 
                self.x[column] = self.x[column].replace(0, mean_val)
        # 標準化
        self.means = self.x.mean(axis=0)
        self.stds = self.x.std(axis=0)
        self.x = (self.x - self.means) / self.stds
        self.x = self.x.to_numpy()
    def get_Process_data(self): #方便維護(ex. 變換)
        return self.x, self.y
if __name__=="__main__": # test
    dataset_path=[
        os.path.join("test_A", "train_data.csv"),
        os.path.join("test_A", "test_data.csv"),
        os.path.join("test_B", "train_data.csv"),
        os.path.join("test_B", "test_data.csv")
    ]
    for path in dataset_path:
        dataset = Dataset(path)
        dataset.Read_data()
        dataset.Process_data()
        x, y = dataset.get_Process_data()

        print(f"路徑: {path}")
        print(f"形狀: {x.shape}, 目標變數形狀: {y.shape}")