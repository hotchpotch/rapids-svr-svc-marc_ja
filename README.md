# rapids-svr-svc-marc_ja

## marc_ja のデータ作成

- https://github.com/yahoojapan/JGLUE#marc-ja

上記方法に沿って行います。以下の例では、`tmp/JGLUE/datasets/marc_ja-v1.1/` に marc_ja のデータを作成しています。

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
ls -alh ls -alh tmp/JGLUE/datasets/marc_ja-v1.1/
```