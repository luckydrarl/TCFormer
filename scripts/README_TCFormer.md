# TCFormer 官方复现指南

## 1. 环境配置

### 创建 conda 环境
```bash
conda create -n tcformer python=3.10 -y
conda activate tcformer
```

### 安装 PyTorch (根据 GPU 选择)
```bash
# CUDA 13.0 (RTX 50 系列 / Blackwell)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# CUDA 12.6 (RTX 30/40 系列 / Ampere/Hopper)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch torchvision
```

### 安装依赖
```bash
pip install -r external/TCFormer/requirements.txt
```

### 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}'); import mne; print(f'MNE: {mne.__version__}'); import moabb; print(f'MOABB: {moabb.__version__}')"
```

## 2. 项目目录结构

```
project/
├── data/BCI2a/              # BCI Competition IV 2a 数据 (MOABB 自动下载到 ~/mne_data/)
├── external/TCFormer/       # TCFormer 官方仓库
├── results/
│   ├── tables/              # 结果 CSV
│   ├── logs/                # 训练日志
│   └── checkpoints/         # 模型权重
└── scripts/                 # 辅助脚本
```

## 3. 数据

### 数据来源
BCI Competition IV 2a 数据集通过 MOABB 自动下载（首次运行时会自动下载到 `~/mne_data/MNE-bnci-data/`）。

### 数据集信息
- 通道数: 22 EEG (去除 3 个 EOG)
- 采样率: 250 Hz
- Trial 长度: 0–4s (cue 后), 即 1000 个时间点
- 类别: 4 (左手, 右手, 双脚, 舌头)
- 被试: 9 人 (A01–A09)
- Session T: 训练 (每个被试 288 trials)
- Session E: 测试 (每个被试 288 trials)

### 数据格式
```
x.shape = [n_trials, 22, 1000]   # [samples, channels, time]
y.shape = [n_trials]              # 0=feet, 1=left_hand, 2=right_hand, 3=tongue
```

### 预处理流程 (官方)
1. 选择 EEG 通道 (22 个)
2. 缩放信号 (×1e6, V → μV)
3. 重采样到 250 Hz
4. 可选: bandpass 滤波 (BCI2a 配置中未启用)
5. 提取 trial 窗口 (0–4s post-cue)
6. Z-score 标准化 (每通道, 仅在训练集上 fit)

## 4. 运行训练

### Subject-Dependent (被试内)
```bash
cd external/TCFormer
python train_pipeline.py --model tcformer --dataset bcic2a --interaug --gpu_id 0 --seed 42
```

### LOSO (Leave-One-Subject-Out)
```bash
cd external/TCFormer
python train_pipeline.py --model tcformer --dataset bcic2a --loso --gpu_id 0 --seed 42
```

### 其他模型
```bash
python train_pipeline.py --model atcnet --dataset bcic2a --interaug --gpu_id 0 --seed 42
python train_pipeline.py --model eegconformer --dataset bcic2a --interaug --gpu_id 0 --seed 42
python train_pipeline.py --model eegtcnet --dataset bcic2a --interaug --gpu_id 0 --seed 42
python train_pipeline.py --model eegnet --dataset bcic2a --interaug --gpu_id 0 --seed 42
```

### 批量运行所有配置
```bash
bash run_all.sh
```

## 5. 查看结果

```bash
# 按被试汇总
python summarize_per_subject.py results/

# 按数据集汇总
python summarize_results.py results/TCFormer/2a
```

## 6. 常见报错与解决

### CUDA error: no kernel image is available for execution on the device
**原因**: PyTorch CUDA 版本不兼容 GPU 计算能力。
- RTX 50 系列 (Blackwell, sm_120) 需要 PyTorch cu130
- RTX 30/40 系列需要 PyTorch cu126
**解决**: 安装与 GPU 匹配的 PyTorch 版本。

### DataLoader worker 崩溃 (Windows)
**原因**: Windows 的 spawn 多进程模式下，每个 worker 都会加载 CUDA DLL，导致虚拟内存不足。
**解决**: 在 configs/tcformer.yaml 的 bcic2a 部分添加 `num_workers: 0`

### prefetch_factor requires num_workers > 0
**原因**: 当 num_workers=0 时，不能设置 prefetch_factor。
**解决**: 已在 datamodules/base.py 中修复：当 num_workers=0 时自动设置 prefetch_factor=None。

### MOABB 下载数据失败
**原因**: 网络代理未设置。
**解决**: 
```bash
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808
```

### numpy 版本冲突
**原因**: moabb 0.5.0 要求 numpy < 2.0。
**解决**: 使用 Python 3.10 环境，requirements.txt 中已固定 numpy==1.26.4。

## 7. 训练参数

| 参数 | Subject-Dependent | LOSO |
|------|-------------------|------|
| max_epochs | 1000 | 125 |
| batch_size | 48 | 48 |
| warmup_epochs | 20 | 3 |
| optimizer | Adam | Adam |
| lr | 0.0009 | 0.0009 |
| weight_decay | 0.001 | 0.001 |

## 8. 结果保存位置

训练结果默认保存在 `external/TCFormer/results/` 下，格式:
```
results/TCFormer_bcic2a_seed-42_aug-True_GPU0_YYYYMMDD_HHMM/
├── checkpoints/
├── confmats/
├── curves/
├── config.yaml
└── summary.csv
```

统一结果表保存在 `results/tables/tcformer_bci2a_results.csv`
