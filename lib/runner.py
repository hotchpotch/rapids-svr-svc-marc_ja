from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import yaml

# add system path
import sys

sys.path.append(str(Path(__file__).parent))


from tfidf_embs import get_tfidf_embs
from transformer_embs import get_transformer_embs
from rapids_svc import train_svc


EMBS_PATH = Path(__file__).parent.parent / "embs"
print(EMBS_PATH)
EMBS_FILES = [f.name for f in EMBS_PATH.glob("*.yaml")]


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
    # if emb_config["type"] == "tfidf":
    #     pass
    print(emb_config)


def run(emb_names: list[str], marc_dir: str, debug: bool = False):
    train_df, valid_df = _get_marc_df(marc_dir)

    if debug:
        train_df = train_df.sample(n=1000, random_state=42)
        valid_df = valid_df.sample(n=200, random_state=42)

    embs = []
    for emb_name in emb_names:
        emb_config = _get_emb_config(emb_name)
        emb = _get_emb(emb_config, train_df, valid_df, debug=debug)
        embs.append(emb)


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
