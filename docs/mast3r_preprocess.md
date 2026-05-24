# MASt3R で 3DGS 用データセットを作る

このメモは、COLMAP の代わりに MASt3R でカメラ姿勢と初期点群を推定し、3D Gaussian Splatting がそのまま読める `images/` と `sparse/0/` を作るための手順です。

## 位置づけ

通常の `convert.py` / `tools/preprocess_with_pycolmap.py` は、COLMAP または pycolmap で以下を作ります。

```text
<dataset>/images/
<dataset>/sparse/0/cameras.bin or cameras.txt
<dataset>/sparse/0/images.bin or images.txt
<dataset>/sparse/0/points3D.bin or points3D.txt
```

`tools/preprocess_with_mast3r.py` は、MASt3R の sparse global alignment から同じ構造を作ります。

```text
<dataset>/input/                  # 元画像コピー
<dataset>/images/                 # 3DGS が読む画像
<dataset>/sparse/0/cameras.txt    # PINHOLE intrinsics
<dataset>/sparse/0/images.txt     # camera poses
<dataset>/sparse/0/points3D.txt   # initial sparse points
<dataset>/sparse/0/points3D.ply   # 3DGS 初回読み込み用PLY
<dataset>/mast3r_cache/           # MASt3R cache
```

## 依存関係

3DGS 本体の環境に加えて、MASt3R とその DUSt3R 依存が必要です。MASt3R は研究コードなので、3DGS の conda 環境へ無理に固定依存として入れず、任意の前処理ツールとして分離しています。

例:

```bash
# 3DGS repo root
source .venv/bin/activate

# 別の作業ディレクトリに MASt3R を取得して editable install
# MASt3R 側の README に従って依存関係を入れてください。
git clone https://github.com/naver/mast3r.git third_party/mast3r
uv pip install --python .venv/bin/python -e third_party/mast3r
```

必要なら MASt3R の checkpoint は Hugging Face から自動取得されます。オフライン環境では、事前に checkpoint をキャッシュしてください。

## 実行例

```bash
.venv/bin/python tools/preprocess_with_mast3r.py \
  --source-images data/my-scene-source/images \
  --dataset-dir data/my-scene-mast3r \
  --model-name naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --device cuda \
  --image-size 512 \
  --scene-graph complete \
  --max-points 300000 \
  --overwrite
```

その後、通常通り 3DGS を学習できます。

```bash
.venv/bin/python train.py \
  -s data/my-scene-mast3r \
  -m output/my-scene-mast3r \
  --disable_viewer \
  --data_device cpu
```

## オプション

- `--source-images`: 入力画像ディレクトリ。
- `--dataset-dir`: 3DGS 互換データセットの出力先。
- `--model-name`: `AsymmetricMASt3R.from_pretrained()` に渡すモデル名。
- `--image-size`: MASt3R 内部推論サイズ。まずは `512` 推奨。
- `--scene-graph`: `dust3r.image_pairs.make_pairs()` に渡す scene graph。画像数が多い場合、`complete` は重いです。
- `--shared-intrinsics`: 全画像で同一内部パラメータを仮定したい場合に指定します。
- `--opt-depth`: MASt3R の depth 最適化を有効化します。
- `--matching-conf-thr`: MASt3R の対応点信頼度しきい値。
- `--max-points`: 3DGS 初期点群へ出す最大点数。`0` で無制限。
- `--overwrite`: 出力先を上書きします。

## 注意点

MASt3R は COLMAP と違って、物理カメラモデル・歪み補正・トラック情報を同じ形で出すわけではありません。このツールは 3DGS が必要とする最小構成として、MASt3R の推定 pose / intrinsics / colored points を COLMAP text 形式に変換します。

`images.txt` の 2D track 行は空にしています。3DGS の通常学習では、登録済みカメラ姿勢と初期点群があればよく、2D観測トラックは使用しません。

品質が悪い場合は、次を試してください。

- 入力画像を増やす、またはブレ・露出差の大きい画像を除外する。
- `--image-size` を上げる。
- `--matching-conf-thr` を下げる/上げる。
- 動画由来なら `tools/extract_video_frames.py` の `--fps` を調整する。
- 通常の pycolmap 前処理結果と MASt3R 前処理結果を比較する。
