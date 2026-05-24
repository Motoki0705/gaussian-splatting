# uv で 3D Gaussian Splatting をセットアップし、前処理から学習とレンダリングまで実行する手順

このメモは、このリポジトリで実際に通した環境構築、画像のみデータセットからの COLMAP 前処理、3000 iteration の学習、学習済みモデルのレンダリングまでを再現するための手順です。

## 実行環境

- OS: Ubuntu 24.04.4 LTS
- Python: uv 管理の CPython 3.11.15
- uv: 0.11.3
- GPU: NVIDIA GeForce RTX 5060 Ti
- Driver: 595.79
- CUDA SDK: 13.0
- PyTorch: 2.12.0+cu130

作業ディレクトリはリポジトリルートです。

```bash
cd /home/kamimura/projects/gaussian-splatting
```

## 1. サブモジュールを取得する

```bash
git submodule update --init --recursive
```

## 2. uv で Python 仮想環境を作成する

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

## 3. Python 依存関係をインストールする

CUDA 13.0 の PyTorch wheel を使います。

```bash
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python plyfile tqdm opencv-python joblib tensorboard pycolmap ninja
```

CUDA 13 / C++20 では `diff-gaussian-rasterization` の一部ヘッダで `uint32_t`, `uint64_t`, `std::uintptr_t` が未定義になるため、先に互換パッチを適用します。

```bash
git -C submodules/diff-gaussian-rasterization apply \
  ../../patches/diff-gaussian-rasterization-cuda13-cstdint.patch
```

3DGS の CUDA 拡張をビルドします。RTX 5060 Ti は compute capability 12.0 として認識されたため、`TORCH_CUDA_ARCH_LIST=12.0` を指定しました。

```bash
TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 \
  uv pip install --python .venv/bin/python --no-build-isolation \
  submodules/simple-knn \
  submodules/diff-gaussian-rasterization \
  submodules/fused-ssim
```

### CUDA 13 向けの互換修正の内容

互換パッチは、以下の include をサブモジュール内のヘッダに追加します。

対象ファイル:

```text
submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
```

追加内容:

```cpp
#include <cstddef>
#include <cstdint>
```

## 4. import と CUDA を確認する

```bash
.venv/bin/python - <<'PY'
import torch
import pycolmap
import diff_gaussian_rasterization
import simple_knn._C
import fused_ssim

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("pycolmap", pycolmap.__version__)
print("3dgs extensions OK")
PY
```

今回の確認結果:

```text
torch 2.12.0+cu130
cuda 13.0
cuda available True
device NVIDIA GeForce RTX 5060 Ti
capability (12, 0)
pycolmap 4.0.4
3dgs extensions OK
```

## 5. 画像データセットを取得する

データセットは COLMAP の South Building dataset を使用しました。

- データセット説明: https://colmap.github.io/datasets.html
- 使用した zip: https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip

```bash
mkdir -p downloads data/south-building-source
wget -O downloads/south-building.zip \
  https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip
unzip -q downloads/south-building.zip -d data/south-building-source
```

zip には sparse reconstruction も含まれていますが、今回の検証ではそれを使わず、`images/` の 128 枚だけを入力にして前処理しました。

## 6. pycolmap で画像から前処理する

### 動画から学習する場合

動画を入力にする場合は、先に ffmpeg でフレーム画像へ変換してから、このセクションの `--source-images` に抽出先ディレクトリを指定します。

```bash
.venv/bin/python tools/extract_video_frames.py \
  --video data/meiji-court-large/VID_20260524_182756335.mp4 \
  --output-dir data/meiji-court-large/video-frames \
  --fps 5 \
  --overwrite
```

`tools/preprocess_with_pycolmap.py` を追加して、以下をまとめて実行できるようにしました。

- 元画像を `input/` にコピー
- SIFT feature extraction
- exhaustive または sequential matching
- incremental mapping
- image undistortion
- 3DGS が読む `images/` と `sparse/0/` の生成

実行コマンド:

```bash
.venv/bin/python tools/preprocess_with_pycolmap.py \
  --source-images data/south-building-source/south-building/images \
  --dataset-dir data/south-building \
  --max-image-size 1600 \
  --matcher sequential \
  --overwrite
```

今回の前処理結果:

```text
registered images: 128 / 128
sparse points: 101345
cameras: 1
camera model after undistortion: PINHOLE
```

確認コマンド:

```bash
.venv/bin/python - <<'PY'
import pycolmap

rec = pycolmap.Reconstruction("data/south-building/sparse/0")
print("images", rec.num_reg_images())
print("points", rec.num_points3D())
print("cameras", rec.num_cameras())
print("models", sorted({cam.model.name for cam in rec.cameras.values()}))
PY
```

## 7. 3000 iteration 学習する

環境構築の疎通確認が目的なので、3000 iteration で実行しました。画像は半解像度 `-r 2`、画像データは CPU 側に保持する `--data_device cpu` を使っています。

```bash
PYTHONUNBUFFERED=1 .venv/bin/python train.py \
  -s data/south-building \
  -m output/south-building-3000 \
  --iterations 3000 \
  --save_iterations 3000 \
  --test_iterations 3000 \
  --disable_viewer \
  -r 2 \
  --data_device cpu
```

今回の学習結果:

```text
Training complete.
iteration: 3000
train L1: 0.06787239834666252
train PSNR: 19.472233390808107
initial points: 101345
trained Gaussians: 356503
```

主な出力:

```text
output/south-building-3000/cfg_args
output/south-building-3000/input.ply
output/south-building-3000/cameras.json
output/south-building-3000/exposure.json
output/south-building-3000/point_cloud/iteration_3000/point_cloud.ply
```

学習済み点群の確認:

```bash
.venv/bin/python - <<'PY'
from plyfile import PlyData

ply = PlyData.read("output/south-building-3000/point_cloud/iteration_3000/point_cloud.ply")
print("vertices", ply["vertex"].count)
PY
```

## 8. 学習済みモデルをレンダリングして保存する

`render.py` は `output/south-building-3000/cfg_args` から学習時の dataset 設定を読みます。今回は `--eval` なしで学習しており test set が空なので、train camera のみをレンダリングしました。

```bash
PYTHONUNBUFFERED=1 .venv/bin/python render.py \
  -m output/south-building-3000 \
  --iteration 3000 \
  --skip_test
```

保存先:

```text
output/south-building-3000/train/ours_3000/renders/
output/south-building-3000/train/ours_3000/gt/
```

今回のレンダリング結果:

```text
rendered PNGs: 128
ground-truth PNGs: 128
image size: 800 x 598
```

確認コマンド:

```bash
find output/south-building-3000/train/ours_3000/renders -maxdepth 1 -type f -name "*.png" | wc -l
find output/south-building-3000/train/ours_3000/gt -maxdepth 1 -type f -name "*.png" | wc -l
```

## 9. レンダリング結果を MP4 にまとめる

PNG の連番を確認しやすいように MP4 も作成しました。

```bash
ffmpeg -y \
  -framerate 24 \
  -i output/south-building-3000/train/ours_3000/renders/%05d.png \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  output/south-building-3000/train/ours_3000/renders.mp4
```

保存先:

```text
output/south-building-3000/train/ours_3000/renders.mp4
```

今回の MP4 は 128 frames、24 fps、約 5.2 秒です。

## 10. 最終成果物

レンダリング画像:

```text
output/south-building-3000/train/ours_3000/renders/00000.png
...
output/south-building-3000/train/ours_3000/renders/00127.png
```

GT 画像:

```text
output/south-building-3000/train/ours_3000/gt/00000.png
...
output/south-building-3000/train/ours_3000/gt/00127.png
```

レンダリング動画:

```text
output/south-building-3000/train/ours_3000/renders.mp4
```

学習済み Gaussian:

```text
output/south-building-3000/point_cloud/iteration_3000/point_cloud.ply
```

## 再実行時の注意

- `.venv` は約 5GB 使用します。
- `downloads/south-building.zip` は約 400MB です。
- `data/south-building` は前処理済みデータとして約 776MB 使用しました。
- `pycolmap` wheel は `COLMAP_build` が `without CUDA` だったため、特徴抽出とマッチングは CPU 実行でした。
- 公式の `convert.py` は外部 `colmap` コマンドを必要としますが、この環境では `colmap` 実行ファイルがなかったため、`pycolmap` で前処理しました。
