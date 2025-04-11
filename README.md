# 檔案說明
- `KNN.py` KNN實現邏輯，要增加其他algos就再新增如 `SVM.py` ...
- `SVM.py` ...
- `dataset.py` 資料處理，可選擇是否featrue selection和是否missing handle
- `main.py` 主要KNN Controler，可彈性變動任何參數輸出不同的實驗結果
- `compare_missing_handle.py` `accuracy_recall_compare.csv` 印出missing handle or not的比較結果，如果要印出其他比較實驗結果(ex. SVM vs. Random Forest ...)可再新增其他.py
- `eulidean.png` `mahattan.png` 比較兩種距離計算公式的結果(沒什麼意義 都用mahattan就好)
- `AfterFeatureSelection.png` 刪掉兩個貢獻較少的特徵的準確率結果
- `EmbeddedRandomForest.py` 用RandomForest做 Feature Importance Analysize (用於特徵選取)
- `rf_feature_importance.png` 比較不同個特徵的重要程度
- `KNN_SVM_Compare.py` `KNN_SVM_Compare.png` 比較KNN和SVM的accuracy & recall

# 主要實驗說明 & 設計
1. 距離計算方式比較 -> Manhattan較穩定
2. 選擇K & C 值方法-**Cross Validation** -> 避免overfitting 或是 underfitting 提升model reliability
3. SVM vs. KNN -> 差不多
4. Missing handling -> 對缺失數據進行補齊 可提高準確/召回率
5. Embedded feature selection -> 以random forest 訓練模型取得個特徵重要程度 用於驗證篩選哪兩個特徵

<!-- 報告就照著主要實驗設計 <-> 圖表寫即可-->
