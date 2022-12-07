"""
tfidf から次元圧縮した特徴量を取得する
"""

import fugashi
import unidic_lite as unidic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

TARGET_POS = ["名詞", "形容詞", "副詞"]


def _create_vectorizer(text, min_df=2):
    vectorizer = TfidfVectorizer(min_df=min_df)
    vectorizer.fit(text)
    return vectorizer


def _create_svd(vect, n_components=1000):
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(vect)
    return svd


def get_tfidf_embs(texts, min_df=2, n_components=1000, target_pos=TARGET_POS):
    tagger = fugashi.Tagger('-d "{}"'.format(unidic.DICDIR))  # type: ignore

    def preprocess_text(text):
        return " ".join(
            [w.surface for w in tagger(text) if w.feature.pos1 in TARGET_POS]
        )

    # 前処理
    texts = [list(map(preprocess_text, text)) for text in texts]
    text = texts[0]
    vectorizer = _create_vectorizer(text, min_df=min_df)
    svd = _create_svd(vectorizer.transform(text), n_components=n_components)
    embs = []
    for text in texts:
        vect = vectorizer.transform(text)
        # print(vect)
        emb = svd.transform(vect)
        # print(emb)
        embs.append(emb)
    return embs


if __name__ == "__main__":
    texts = [
        [
            "今日はいい天気ですね。今日は温かいですね。",
            "今日は雨が降りそうですね。",
            "今日はいい天気ですね。今日は温かいですね。今晩も温かいですね。",
        ],
        [
            "今日はいい天気ですね",
            "今日は雨が降りそうですね",
        ],
    ]
    embs = get_tfidf_embs(texts, min_df=1, n_components=2)
    print(embs[0].shape)
    print(embs[1].shape)
