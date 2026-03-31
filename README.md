# Animal Metrology

本專案旨在將**圖像分割（Image Segmentation）模型**的輸出，應用於實際的**測量學（Metrology）任務**。

使用COCO資料集，從中篩選出「包含兩隻（含）以上動物」的圖片，並透過圖像模型進行動物輪廓與雙眼位置偵測，再進一步進行眼睛距離量測。

### 任務目標
1. 從COCO資料集中篩選出包含兩隻（含）以上動物的圖片  
2. 使用圖像模型框出：
   - 每隻動物的輪廓
   - 每隻動物的眼睛位置  
3. 進行以下測量：
   - 每隻動物的雙眼距離
   - 任意兩隻動物的右眼之間距離  


## 技術棧

### Backend / Processing
- Docker / Docker Compose
- Python 3.10
- OpenCV
- NumPy
- Ultralytics
- Pycocotools

### AI / Computer Vision
- YOLO模型 (Segmentation) — 用於動物實體偵測
- YOLO模型 (Pose Estimation) — 用於眼睛位置偵測
- COCO Dataset — 標準資料集（含多類動物）


## 檔案結構

```
animal-metrology/
│
├── build/                    # 環境建置與部署相關設定
│   ├── dockerfile            # Docker映像檔建置腳本
│   └── requirements.txt      # Python套件列表
│
├── data/                     # 存放資料
│   ├── coco/                 # COCO資料集
│   └── output/               # 模型推論結果與後處理輸出
│       ├── sample            # 匯出範例檔案
│           ├── sample_animals.csv           # 每隻動物的雙眼距離CSV檔案
│           └── sample_inter_distance.csv    # 任意兩隻動物的右眼之間距離CSV檔案 
│       ├── selected_image    # 篩選後的圖片
│       └── result            # 結果匯出檔案
│
├── document/                 # 說明文件檔案
│
├── model/                    # 模型權重檔案        
│
├── src/                      # 核心程式碼
│   ├── utils/                # 工具模組
│   │   ├── coco_utils.py     # COCO資料解析與處理
│   │   ├── distance.py       # 距離計算
│   │   ├── download.py       # 資料下載
│   │   ├── draw.py           # 繪圖、視覺化
│   │   ├── metrics.py        # 評估指標計算
│   │   └── yolo_utils.py     # YOLO模型相關操作
│   │
│   ├── config.py             # 程式環境變數與全域設定
│   ├── detector.py           # 偵測
│   ├── download_dataset.py   # 下載COCO資料集
│   ├── filter_data.py        # 篩選圖片
│   └── predict.py            # 模型推論 + 距離計算 + 結果輸出
│
├── .env.example              # 環境變數範本
├── docker-compose.yml        # Docker容器部署設定
└── README.md                 # 專案說明文件

```


## 本地運行步驟

### 一、啟動服務

1. 切換至`animal-metrology`資料夾:
```bash
cd animal-metrology
```
<br>

2. 設定環境變數

複製`.env.example`，複製成檔名為`.env`的檔案：
```bash
cp .env.example .env
```

根據需求修改`.env`內的環境變數設定，可使用機器上支援的任何編輯器，以vim與nano為例：
- vim:
```bash
vim .env
```

- nano:
```bash
nano .env
```
<br>

3. 啟動服務:
```bash
docker compose up -d
```

### 二、開始使用

1. 下載 COCO Dataset

```bash
docker exec animal-metrology python3 download_dataset.py
```

下載後的dataset存放在`data/coco/`資料夾。
<br>

2. 篩選符合條件的圖片（≥2 隻動物的圖片）並匯出結果

```bash
docker exec animal-metrology python3 filter_data.py
```

篩選後的圖片存放在`data/output/selected_images/`資料夾。
<br>

3. 執行模型推論 (動物實體與眼睛位置偵測、眼睛距離計算) 並匯出結果

```bash
docker exec animal-metrology python3 predict.py
```

匯出的結果存放在`data/output/result/`資料夾。結果包含：
* 框選後的圖片 (格式: <圖片ID>.jpg)
* 距離計算結果 CSV:
    - 每隻動物的雙眼距離CSV檔案（格式: <圖片ID>_animals.csv）
    - 任意兩隻動物的右眼之間距離CSV檔案（格式: <圖片ID>_inter_distance.csv）

距離計算結果CSV範例檔存放在`data/output/sample/`資料夾。


### 三、關閉服務
切換至`animal-metrology`資料夾，執行指令:
```bash
docker compose down
```

---

## 模型選擇與原因

### 1.動物輪廓分割（Segmentation）

#### 模型選擇：
- YOLO Segmentation 模型 (yolo26x-seg.pt)

#### 選擇原因：
根據 Ultralytics 官方文件（[YOLO Segmentation](https://docs.ultralytics.com/tasks/segment/)）中的模型效能比較，可以觀察到目前最新一代模型 YOLOv26 模型效果優良。根據效能比較表，可發現 YOLO26x 模型在 mAP（mean Average Precision）、推論速度指標上表現最佳，因此本專案選擇此模型作為主要的動物輪廓偵測模型。

Ultralytics segmentation 模型效能比較圖：
![Segmentation Benchmark](document/best_seg_model.jpeg)

### 2.動物眼睛偵測（Eye Detection）

#### 模型選擇：
- YOLO Pose Estimation 模型 (yolo26x-pos.pt)

#### 選擇原因：
根據相關實作與技術文章（如 [OpenCV Eye Tracking: Step-By-Step With Code](https://medium.com/@amit25173/opencv-eye-tracking-aeb4f1b46aa3)）指出，使用深度學習模型通常偵測精準度較高。因此，本專案採用 YOLO Pose Estimation 模型 進行眼睛關鍵點偵測。

另外，根據 Ultralytics 官方 Pose Estimation 文件（[YOLO Pose Estimation](https://docs.ultralytics.com/tasks/pose/)），YOLO Pose 模型可輸出動物的關鍵點（keypoints），包含：左眼（left eye）與右眼（right eye），因此決定使用此模型。在模型效能比較表中，YOLO26x 模型在 mAP（mean Average Precision）、推論速度指標上表現最佳，因此本專案選擇此模型作為主要的動物眼睛偵測模型。

Ultralytics pose estimation 模型效能比較圖：
![Pose Estimation Benchmark](document/best_pe_model.jpeg)


## 測量方法（Metrology）

### 1️.雙眼距離（同一動物）

假設左右眼座標為：
```
L = (x1, y1)
R = (x2, y2)
```

使用歐幾里得距離公式計算距離：
```
d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

### 2️.兩隻動物右眼距離
假設兩隻動物之右眼座標為：
```
R1 = (x1, y1)
R2 = (x2, y2)
```

與計算雙眼距離方式相同，同樣使用歐幾里得距離公式計算距離：
```
d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

### 3.關鍵點（Keypoint）取得方式
* 使用 segmentation 模型找出動物區域（bounding box）
* 使用 pos estimation 模型，針對動物區域圖片進行 keypoint detection
* 從 pos estimation 模型推論結果取得雙眼keypoint座標


## 模型效果驗證方法

### 1.動物輪廓分割（Segmentation）
使用 **IoU（Intersection over Union）** 作為評估指標：
```
IoU = Area of Overlap / Area of Union
```

由於COCO dataset中包含人工標記的segmentation資訊，因此使用此資訊作為正確答案，並使用模型推論後得到的segmentation資訊作為預測結果，接著使用正確答案與預測結果計算IoU，來評估segmentation的準確度。

### 2.動物眼睛偵測（Eye Detection）
在此專案中，由於選擇的YOLO pose estimation模型主要是針對人體關鍵點進行訓練，因此標記資料中並沒有動物（排除人類）的關鍵點標記資料，導致無法計算偵測準確度。

未來有三種驗證方法：
1. 人工標註驗證
   * 手動標記眼睛位置作為 ground truth
   * 比較預測距離誤差

2. 視覺化檢查
   * 繪製眼睛點位與連線
   * 確認幾何合理性

3. 統計分析
   * 平均誤差（MAE / RMSE）
   * 分布圖（error distribution）


## 未來優化方向
由於目前選擇的YOLO pose estimation模型主要是針對人體關鍵點進行訓練，因此針對動物圖片預測的眼睛定位準確度差，未來可尋找使用針對動物圖片進行訓練的 pose estimation 模型，或是自行標記資料並訓練，以提升眼睛定位準確度。


