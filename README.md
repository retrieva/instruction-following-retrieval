# 指示追従検索

## 目次
* [はじめに](#はじめに)
* [検証した手法](#検証した手法)
* [実験設定](#実験設定)
* [訓練](#訓練)
* [評価](#評価)
* [ディレクトリ構成](#ディレクトリ構成)
* [おわりに](#おわりに)

## はじめに
近年、クエリだけでなくユーザの嗜好や意図を反映した指示文に基づいて検索する指示追従検索に関する研究が注目を集めています。
従来の検索では、目的の文書が検索できるまでクエリを変更しながら繰り返し検索する必要がありますが、指示追従検索では、指示文を付与することでユーザの目的に沿った柔軟な検索を可能とします。
今回は、指示追従検索の性能改善に向けて2種類の重み付き対照学習とマージンロスの手法について紹介します。

## 検証した手法
### 重み付き対照学習（手法1）
正例文書に対して、入力テキストの関連度を考慮した重み付き対照学習です。
重み付き対照学習(手法1)では、正例文書に対して「クエリ・指示文・クエリ+指示文」の3つを正解ラベルとします。
[図1](#fig-1)のように正例文書と入力テキストの関連度を重みとし、損失に掛け合わせします。
この時の重みは、教師ありSimCSEで用いた入力テキストと正例文書のコサイン類似度です。
<p align="center" id="fig-1">
  <img src="https://github.com/retrieva/2025_internship/blob/main/images/weighted_v1.png" width="60%" alt="重み付き対照学習（手法1）">
  <figcaption><p align="center">図1：重み付き対照学習（手法1）の概要図</p></figcaption>
</p>

### 重み付き対照学習（手法2）
入力テキストに対して、どの程度正例文書が関連しているのかを基に訓練する重み付き対照学習です。
通常の対照学習では、「クエリ+指示文と文書」のみを対象としていましたが、クエリや指示文の単体だけでも正例文書に関連しています。
そこで、クエリや指示文にも文書との関連性を考慮するため、[図2](#fig-2)のように「クエリ・指示文・クエリ+指示文」の関連度に応じた重み付き対照学習の手法を実装します。
重み付き対照学習（手法2）における重みは、手法1と同様に教師ありSimCSEを用いて、各入力テキストと文書のコサイン類似度を計算し、各入力テキストごとに正規化することで算出します。
<p align="center" id="fig-2">
  <img src="https://github.com/retrieva/2025_internship/blob/main/images/weighted_v2.png" width="55%" alt="重み付き対照学習（手法2）">
  <figcaption><p align="center">図2：重み付き対照学習（手法2）の概要図</p></figcaption>
</p>

### マージンロス
教師ありSimCSEを用いて、クエリ+指示文と負例文書の類似度が高い場合、マージンの値が大きくなるマージンロスです。
実験では、 $\alpha = 0.4, \ \beta = 0.1$としています。
```math
L = max\{0, m(x, p^-) + sim(x, p^-)-sim(x,p^+)\}
```
```math
m(x,p^-) = \alpha *\tau(x, p^-)+\beta
```

## 実験設定
訓練モデル：[meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)<br>
類似度推定モデル：[princeton-nlp/sup-simcse-bert-large-uncased](https://huggingface.co/princeton-nlp/sup-simcse-bert-large-uncased)<br>
訓練用データセット：[InF-IR/InF-IR](https://huggingface.co/datasets/InF-IR/InF-IR)<br>
評価用データセット：[FollowIR](https://github.com/orionw/FollowIR)<br>


訓練用データセットには、MSMARCOを基にクエリと指示文、正例文書、負例文書から構成された[InF-IR/InF-IR](https://huggingface.co/datasets/InF-IR/InF-IR)を使用しています。
また、指示追従検索のベンチマークとしてRobust2004、Core2017、News2021の3つのデータセットから構成された[FollowIR](https://github.com/orionw/FollowIR)を評価データセットとして使用しています。
評価指標は、nDCG\@5およびp-MRRです。

## 訓練
以下は、重み付き対照学習（手法2）の訓練の実行例です。

```bash
SAVE_MODEL="./save_model/weighted_contrastive_v2"

python ./src/train.py \
 loss.contrastive_loss=false \
 loss.weighted_contrastive_loss_v1=false \
 loss.weighted_contrastive_loss_v2=true \
 loss.margin_loss=false \
 training.output_dir=${SAVE_MODEL} 
```

## 評価
評価時には、[Massive Text Embedding Benchmark](https://github.com/embeddings-benchmark/mteb)のパッケージを使用し、訓練済みモデルをFollowIRで評価しました。
```bash
RESULT_PATH="./results/weighted_contrastive_v2"

python ./src/mteb_eval_custom.py \
 --base_model_name_or_path "meta-llama/Llama-3.2-1B-Instruct"\
 --peft_model_name_or_path $(ls -td ${SAVE_MODEL}/checkpoint-* | head -1) \
 --output_dir ${RESULT_PATH}
```

## ディレクトリ構成
ディレクトリ構造は以下の通りになっており、script/配下で訓練および評価が実行可能です。

```
.
├── pyproject.toml
├── script
├── src
│   ├── config
│   │   ├── config.yaml
│   │   ├── config_multi.yaml
│   │   └── ds_config.json
│   ├── dataset
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── msmarco.py
│   ├── evaluation
│   │   └── early_stop_callback.py
│   ├── loss
│   │   ├── contrastive_loss.py
│   │   ├── loss_utils.py
│   │   ├── margin_loss.py
│   │   ├── tau_module.py
│   │   └── weighted_contrastive_loss.py
│   ├── models
│   │   ├── bidirectional_llama.py
│   │   ├── instract_ir_model.py
│   │   └── utils.py
│   ├── mteb_eval_custom.py
│   ├── multi-gpu-train.py
│   └── train.py
└── uv.lock
```

## おわりに
株式会社レトリバの夏季インターンシップに参加させていただき、指示追従検索に関する研究に取り組みました。
今回は、2種類の対照学習手法とマージンロスの手法を実装しています。
株式会社レトリバの技術ブログにも紹介してありますので、ご確認ください。

