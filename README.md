# 使用 Google Colab GPU 訓練神經網路：MNIST 手寫數字分類

本專案依據知乎文章〈完整的動手指南，可在 Google Colab GPU 上訓練你的神經網絡模型〉整理，使用 PyTorch 建立簡單的全連接神經網路，並以 MNIST 手寫數字資料集示範 GPU / CPU 訓練流程。

> 原文發布日期：2021-06-09  
> 原文作者：安大叔  
> 原文連結：https://zhuanlan.zhihu.com/p/379291795

## 目錄

- [專案摘要](#專案摘要)
- [研究背景與動機](#研究背景與動機)
- [核心觀念](#核心觀念)
- [實作流程](#實作流程)
- [模型架構](#模型架構)
- [執行環境](#執行環境)
- [如何執行](#如何執行)
- [實驗結果與觀察](#實驗結果與觀察)
- [限制與注意事項](#限制與注意事項)
- [結論](#結論)
- [專案結構](#專案結構)
- [參考資料](#參考資料)

## 專案摘要

本專案示範如何使用 Google Colab 提供的運算環境，透過 PyTorch 在 GPU（若可用）上訓練神經網路。實驗任務為 MNIST 手寫數字分類：輸入 28 × 28 灰階影像，輸出 0 至 9 共 10 個類別之一。

Notebook 實作裝置自動選擇機制：若 CUDA GPU 可用便使用 GPU，否則退回 CPU。這使同一份程式能在不同硬體環境執行。

## 研究背景與動機

神經網路訓練包含大量矩陣運算，尤其是神經網路中的矩陣乘法與反向傳播。文章指出，GPU 具有大量運算核心與較高的記憶體頻寬，適合處理可平行化的計算，因此在許多深度學習工作負載中能縮短訓練時間。

文章以 VGG16 等具有大量參數的模型說明 CPU 訓練大型模型可能耗費大量時間，並將焦點轉向以 Google Colab GPU 訓練較簡單的 MNIST 分類模型。

## 核心觀念

### CPU 與 GPU 的分工

- **CPU**：負責一般程式控制、資料準備及各種通用運算。
- **GPU**：以大量平行運算核心處理適合平行化的計算，例如神經網路常見的矩陣運算。
- **CUDA**：NVIDIA GPU 的平行運算平台。PyTorch 提供較方便的介面，讓使用者能將模型與 Tensor 放到 CUDA 裝置上。

GPU 並非在所有任務都優於 CPU。模型規模、資料量、硬體規格及 CPU-GPU 資料傳輸成本，都會影響實際速度。

### MNIST 分類任務

MNIST 是手寫數字影像資料集。每筆影像為 28 × 28 灰階像素，標籤為 0 至 9。本文實作將 60,000 筆訓練資料切分為：

| 資料集 | 數量 |
|---|---:|
| 訓練集 | 50,000 |
| 驗證集 | 10,000 |

## 實作流程

本專案依照原文主要步驟整理，並補齊 PDF 中未完整呈現的程式碼細節。

1. **設定 Google Colab 執行階段**：建議選擇 GPU。
2. **載入套件**：使用 PyTorch、TorchVision、NumPy、Matplotlib。
3. **建立訓練與驗證資料集**：下載 MNIST，並保留 10,000 筆作為驗證集。
4. **批次載入資料**：訓練 batch size 設為 128，驗證 batch size 設為 256；CUDA 環境可啟用 `pin_memory`。
5. **建立模型類別**：定義兩層全連接網路，隱藏層使用 ReLU。
6. **選擇運算裝置**：以 `torch.cuda.is_available()` 判斷是否使用 CUDA。
7. **建立資料搬移函式**：將 Tensor 移至指定裝置。
8. **建立 DeviceDataLoader**：每次取出 batch 時自動將資料移至 GPU 或 CPU。
9. **定義訓練與驗證函式**：使用交叉熵損失與 SGD 優化器。
10. **建立模型並搬移裝置**：確保模型與資料位於相同裝置。
11. **執行訓練與驗證**：記錄訓練損失、驗證損失及驗證準確率。

## 模型架構

本專案採用簡單的多層感知器（MLP）：

```text
輸入影像 (1 × 28 × 28)
        |
   Flatten (784)
        |
全連接層 (784 → 32)
        |
      ReLU
        |
全連接層 (32 → 10)
        |
   10 類別 logits
```

- 輸入維度：784
- 隱藏層：32 個神經元
- 輸出維度：10
- 損失函數：Cross Entropy
- 優化器：SGD
- Notebook 預設訓練設定：5 epochs、learning rate 0.1

## 執行環境

- Python 3
- PyTorch
- TorchVision
- NumPy
- Matplotlib
- Jupyter Notebook 或 Google Colab

安裝依賴套件：

```bash
pip install -r requirements.txt
```

## 如何執行

### 方法一：Google Colab（建議）

1. 將 `analysis.ipynb` 上傳至 Google Colab。
2. 選擇「執行階段」→「變更執行階段類型」。
3. 若可選，將硬體加速器設定為 GPU。
4. 依序執行所有儲存格。
5. 第一次執行時會自動下載 MNIST 資料集。

Colab 的 GPU 型號、使用額度與工作階段時間依 Google 當下政策及資源供應而變動；不可假設固定提供特定 GPU 或固定 12 小時連線。

### 方法二：本機 Jupyter

```bash
pip install -r requirements.txt
jupyter notebook
```

開啟 `analysis.ipynb` 並依序執行。若本機 PyTorch 未安裝 CUDA 版本，程式會自動使用 CPU。

## 實驗結果與觀察

Notebook 會在每個 epoch 輸出：

- `train_loss`：訓練集平均損失。
- `val_loss`：驗證集平均損失。
- `val_acc`：驗證集平均分類準確率。

並繪製訓練／驗證損失曲線與驗證準確率曲線，也會顯示部分影像的真實標籤與模型預測。

**請以實際執行 Notebook 所得到的數值與圖表作為實驗結果。** 本 README 不預先填入未經執行驗證的準確率或訓練時間，因為結果會受到隨機初始化、軟體版本與硬體環境影響。

## 限制與注意事項

1. 本專案使用的是入門級 MLP，並非 VGG16 等大型 CNN。
2. GPU 是否可用須以 `torch.cuda.is_available()` 的執行結果為準。
3. GPU 不保證在所有情境都比 CPU 快；小型模型可能受到資料傳輸與啟動成本影響。
4. Notebook 加入了原文 PDF 未完整呈現的必要程式碼，例如驗證準確率計算及訓練／評估模式切換。
5. `num_workers` 在不同作業系統與執行環境可能需要調整；若資料載入發生問題，可先設為 `0`。

## 結論

本專案重現了文章介紹的基本 GPU 訓練工作流程：準備資料、建立 DataLoader、定義模型、選擇裝置、搬移資料與模型，最後進行訓練和驗證。GPU 的大量平行運算能力能協助加速適合的深度學習工作，但 CPU 仍在資料準備與程式控制等環節扮演重要角色。

文章最後也提到 SLIDE（Sub-Linear Deep Learning Engine），其研究方向是減少反向傳播中的部分不必要計算，並探索不依賴硬體加速器、使用 CPU 訓練深度學習模型的可能性。

## 專案結構

```text
mnist_gpu_training_project/
├── README.md
├── analysis.ipynb
└── requirements.txt
```

## 參考資料

1. 安大叔，〈完整的動手指南，可在 Google Colab GPU 上訓練你的神經網絡模型〉，知乎專欄，2021-06-09。https://zhuanlan.zhihu.com/p/379291795
2. PyTorch 官方文件：https://pytorch.org/docs/stable/index.html
3. TorchVision 官方文件：https://pytorch.org/vision/stable/index.html
