# rapids-svr-svc-marc_ja

各種モデルをfinetune無しにRAPIDS SVCで学習した[JGLUE](https://github.com/yahoojapan/JGLUE)のMARC-ja の評価結果とその実装。日本語での解説記事は[こちら](https://secon.dev/entry/2022/12/13/090000-rapids-svr-svc-marc-ja/)。

- https://secon.dev/entry/2022/12/13/090000-rapids-svr-svc-marc-ja/

## スコア

Dev データの正解率

| 設定名 | acc |
| --- | --- |
| bert-base-ja-v2-mean | 0.93244 |
| bert-base-ja-v2-cls | 0.92766 |
| rinna-ja-roberta-base-mean | 0.93314 |
| rinna-ja-roberta-base-cls | 0.92059 |
| tfidf | 0.89247 |
| bert-base-ja-sentiment-mean | 0.93544 |
| bert-base-ja-sentiment-cls | 0.93244 |
| 上記7つの結合特徴量 | 0.94323 |

## MARC_ja のデータ作成

- https://github.com/yahoojapan/JGLUE#marc-ja

上記方法に沿って行う。以下の例では、`tmp/JGLUE/datasets/marc_ja-v1.1/` に marc_ja のデータを作成している。

```
mkdir -p tmp
cd tmp
git clone https://github.com/yahoojapan/JGLUE
cd JGLUE
pip install -r preprocess/requirements.txt
cd preprocess/marc-ja/scripts
wget https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_JP_v1_00.tsv.gz
gzip -dc ./amazon_reviews_multilingual_JP_v1_00.tsv.gz | \
  python marc-ja.py \
         --positive-negative \
         --output-dir ../../../datasets/marc_ja-v1.1 \
         --max-char-length 500 \
         --filter-review-id-list-valid ../data/filter_review_id_list/valid.txt \
         --label-conv-review-id-list-valid ../data/label_conv_review_id_list/valid.txt
cd ../../../../../
ls -alh tmp/JGLUE/datasets/marc_ja-v1.1/
```

## 各種ライブラリのインストール

### RAPIDS

- https://rapids.ai/start.html

### ライブラリ各種

```
pip install -r requirements.txt
```

## 実行

```
# -d をつけた debug 実行では少データで実行する
# python lib/runner.py -d bert-base-ja-v2-cls
python lib/runner.py bert-base-ja-v2-cls
# なお、引数の名前は、`./embs/*.yaml` の設定名である。
```

```
[load cache] tmp/embs_cache/bert-base-ja-v2-cls.pkl.gz
shape: (187528, 768) (5654, 768)
concat embs: (187528, 768) (5654, 768)
[train svc]
svc exec time: 22.12 sec
==================================================
bert-base-ja-v2-cls
valid acc score: 0.927661832331093
==================================================
              precision    recall  f1-score   support

    positive    0.89788   0.56691   0.69500       822
    negative    0.93067   0.98903   0.95896      4832

    accuracy                        0.92766      5654
   macro avg    0.91428   0.77797   0.82698      5654
weighted avg    0.92590   0.92766   0.92059      5654
```

複数モデルの特徴量を結合しての実行例

```
$ python lib/runner.py bert-base-ja-v2-cls bert-base-ja-v2-mean rinna-ja-roberta-base-cls rinna-ja-roberta-base-mean tfidf bert-base-ja-sentiment-cls bert-base-ja-sentiment-mean
[load cache] tmp/embs_cache/bert-base-ja-v2-cls.pkl.gz
shape: (187528, 768) (5654, 768)
(略)
[load cache] tmp/embs_cache/bert-base-ja-sentiment-mean.pkl.gz
shape: (187528, 768) (5654, 768)
concat embs: (187528, 5608) (5654, 5608)
[train svc]
svc exec time: 89.47 sec
==================================================
bert-base-ja-v2-cls + bert-base-ja-v2-mean + rinna-ja-roberta-base-cls + rinna-ja-roberta-base-mean + tfidf + bert-base-ja-sentiment-cls + bert-base-ja-sentiment-mean
valid acc score: 0.9432260346657234
==================================================
              precision    recall  f1-score   support

    positive    0.93717   0.65328   0.76989       822
    negative    0.94391   0.99255   0.96762      4832

    accuracy                        0.94323      5654
   macro avg    0.94054   0.82292   0.86876      5654
weighted avg    0.94293   0.94323   0.93887      5654
```
