
# NeuCF推薦系統專案

矩陣分解推薦系統項目

專案概述
本專案採用矩陣分解技術來建立推薦系統模型，目的是提升電影推薦的準確度。整個開發流程使用 PyTorch 框架，包含了數據前處理、模型訓練、參數優化及模型效能測試等階段。

# 架構圖

![NeuCF (1)](https://github.com/9270123a/MatrixFactorization/assets/157206678/ba3f26c4-8005-44b2-b46e-b8a9b4ac6938)


## 安裝步驟

為避免依賴衝突，建議在虛擬環境中進行安裝和運行。可以透過以下命令建立並啟用虛擬環境：

```bash
python3 -m venv venv
source venv/bin/activate  # 針對 Linux 或 macOS
venv\Scripts\activate  # 針對 Windows
```
安裝必要的依賴包：
```bash
pip install torch==1.7.1 pandas==1.1.5 scikit-learn==0.23.2 matplotlib==3.3.3
```


## 環境配置

Python 版本：3.6 及以上

PyTorch 版本：1.7.1 或更高

Pandas 版本：1.1.5 或更高

Scikit-learn 版本：0.23.2 或更高

Matplotlib 版本：3.3.3 或更高（選用，用於製圖）
## 檔案結構說明


preprocessing.py：負責加載、清潔和準備數據的數據前處理腳本。

MatrixFactorization_Train.py：主要的矩陣分解模型訓練腳本。

MatrixFactorizationGridSearch_FingBestPar.py：用於進行網格搜索以尋找最佳模型參數的腳本。

TestMatrix.py：用於測試模型性能的腳本。

## 使用指南


1.數據前處理：先執行 preprocessing.py 以準備訓練和測試數據集。

```bash
python preprocessing.py
```
2.模型訓練：透過執行 MatrixFactorization_Train.py 進行模型訓練。您可以調整腳本中的參數以更改模型配置。
```bash
python MatrixFactorization_Train.py
```
3.參數優化：運行 MatrixFactorizationGridSearch_FingBestPar.py，以利用網格搜索找出最優模型參數。
```bash
python MatrixFactorizationGridSearch_FingBestPar.py
```

4.模型測試：最後，使用 TestMatrix.py 來測試模型的效能，並進行性能評估。
```bash
python TestMatrix.py
```

## 貢獻準則
我們歡迎並鼓勵社群貢獻本專案。若您有任何改善建議或想增加新功能，請透過 Issue 或 Pull Request 提出。






