from __future__ import annotations
import argparse
from pathlib import Path
import joblib

import pandas as pd
import numpy as np
import yaml
import time

from sklearn.metrics import classification_report, accuracy_score

# add system path
import sys

sys.path.append(str(Path(__file__).parent))


from tfidf_embs import get_tfidf_embs
from transformer_embs import get_transformer_embs
from rapids_svc import train_svc


EMBS_PATH = Path(__file__).parent.parent / "embs"
EMBS_FILES = [f.name for f in EMBS_PATH.glob("*.yaml")]
CACHE_PATH = Path(__file__).parent.parent / "tmp/embs_cache"
if not CACHE_PATH.exists():
    CACHE_PATH.mkdir(parents=True)


def _get_emb_config(emb_name: str) -> dict[str, object]:
    emb_path = EMBS_PATH / f"{emb_name}.yaml"
    with open(emb_path) as f:
        emb_data = yaml.safe_load(f)
    return emb_data


def _get_marc_df(marc_dir: str):
    marc_path = Path(marc_dir)
    if not marc_path.exists():
        raise ValueError(f"marc_ja のデータディレクトリ '{marc_path}' が存在しません")
    train_df = pd.read_json(marc_path / "train-v1.0.json", lines=True)
    valid_df = pd.read_json(marc_path / "valid-v1.0.json", lines=True)
    train_df["label"] = train_df.label.astype("category")  # .cat.codes
    valid_df["label"] = valid_df.label.astype("category")  # .cat.codes
    return train_df, valid_df


def _get_emb(
    emb_config: dict[str, object],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    debug: bool = False,
):
    train_texts = train_df.sentence.tolist()  # type: ignore
    valid_texts = valid_df.sentence.tolist()  # type: ignore
    config = dict(emb_config)
    type = config.pop("type")
    if type == "tfidf":
        embs = get_tfidf_embs([train_texts, valid_texts], **config)
        return embs
    elif type == "transformer":
        model_name = config.pop("model_name")
        tokenizer_class_name = config.pop("tokenizer", None)
        embs = get_transformer_embs(
            [train_texts, valid_texts],
            model_name=model_name,  # type: ignore
            tokenizer_class_name=tokenizer_class_name,  # type: ignore
            **config,
        )
        return embs
    else:
        raise ValueError(f"'{type}' はありません")


def _get_emb_with_cache(
    cache_key: str,
    emb_config: dict[str, object],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    debug: bool = False,
):
    if debug:
        cache_key = f"debug_{cache_key}"
    cache_path = CACHE_PATH / f"{cache_key}.pkl.gz"
    if cache_path.exists():
        print(f"[load cache] {cache_path}")
        embs = joblib.load(cache_path)
        print("shape:", embs[0].shape, embs[1].shape)
        return embs
    else:
        print(f"[create cache] {cache_path}")
        start_time = time.time()
        embs = _get_emb(emb_config, train_df, valid_df, debug=debug)
        joblib.dump(embs, cache_path)
        end_time = time.time()
        print(f"exec time: {end_time - start_time:.2f} sec")
        print("shape:", embs[0].shape, embs[1].shape)
        return embs


def run(emb_names: list[str], marc_dir: str, debug: bool = False):
    train_df, valid_df = _get_marc_df(marc_dir)

    if debug:
        train_df = train_df.sample(n=1000, random_state=42)
        valid_df = valid_df.sample(n=200, random_state=42)
        print("[debug] shape size:", train_df.shape, valid_df.shape)

    train_embs = []
    valid_embs = []
    for emb_name in emb_names:
        emb_config = _get_emb_config(emb_name)
        train_emb, valid_emb = _get_emb_with_cache(
            emb_name, emb_config, train_df, valid_df, debug=debug
        )
        train_embs.append(train_emb)
        valid_embs.append(valid_emb)

    train_embs = np.concatenate(train_embs, axis=1).astype("float32")
    valid_embs = np.concatenate(valid_embs, axis=1).astype("float32")
    print("concat embs:", train_embs.shape, valid_embs.shape)

    print("[train svc]")
    start_svc = time.time()
    svc = train_svc(train_embs, train_df.label.cat.codes)  # type: ignore
    valid_preds = svc.predict(valid_embs)
    end_svc = time.time()
    print(f"svc exec time: {end_svc - start_svc:.2f} sec")
    del svc  # for free

    print("=" * 50)
    if debug:
        print("[DEBUG]")
    print(" + ".join(emb_names))
    # acc = accuracy_score(valid_df.label.cat.codes, valid_preds)
    acc = (valid_df.label.cat.codes == valid_preds).sum() / len(valid_preds)
    print("valid acc score:", acc)
    print("=" * 50)

    report_text = classification_report(
        valid_df.label.cat.codes,
        valid_preds,
        digits=5,
        target_names=["positive", "negative"],
    )
    print(report_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "embs",
        type=str,
        help="embs以下の名前。複数指定した場合、複数の特徴量を使う。 e.g.: bert-base-ja-v2-cls rinna-ja-roberta-base-mean",
        nargs="+",
    )
    parser.add_argument(
        "-m",
        "--marc_dir",
        type=str,
        default="tmp/JGLUE/datasets/marc_ja-v1.1/",
        help="marc_ja のデータ置かれているディレクトリ",
    )

    parser.add_argument(
        "-d", "--debug", action="store_true", help="debug, debug時は少量のデータを使う"
    )
    args = parser.parse_args()
    # print(args.embs, args.debug)
    run(emb_names=args.embs, debug=args.debug, marc_dir=args.marc_dir)
