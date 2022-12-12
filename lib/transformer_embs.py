"""
huggingface transformer の embedding を取得する
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import logging, AutoTokenizer, AutoModel, DataCollatorWithPadding  # type: ignore
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

logging.set_verbosity_error()


class MeanPooling(nn.Module):
    def __init__(self, eps=1e-6):
        super(MeanPooling, self).__init__()
        self.eps = eps

    def forward(
        self, outputs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden_state = outputs[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=self.eps)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class ClsPooling(nn.Module):
    # 実際は Pooling ではなくただの CLS を取り出しているだけなので、このクラス名は良くない…
    def __init__(self):
        super(ClsPooling, self).__init__()

    def forward(
        self, outputs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden_state = outputs[0]
        return last_hidden_state[:, 0, :]


POOLING_CLASSES = {
    "mean": MeanPooling,
    "cls": ClsPooling,
}


class TransformerEmbsModel(torch.nn.Module):
    def __init__(self, model_name: str, pooling: str = "mean"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.pool = POOLING_CLASSES[pooling]()

    def feature(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**inputs)
        sentence_embeddings = self.pool(outputs, inputs["attention_mask"])
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings.squeeze(0)
        return sentence_embeddings

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        embs = self.feature(inputs)
        return embs


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def prepare_input(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs

    def __getitem__(self, item):
        text = self.texts[item]
        try:
            inputs = self.prepare_input(text)
            return inputs
        except (ValueError, IndexError):
            # jumanppのtokenizerで特定の文章でエラーが出るので、その対策
            orig_text = "" + text
            while True:
                try:
                    inputs = self.prepare_input(text)
                    return inputs
                except (ValueError, IndexError):
                    min_text_len = int(len(text) * 0.9)
                    text = text[:min_text_len]
                    if len(text) == 0:
                        break
            raise ValueError(f"cannot tokenize: {orig_text}")


@torch.inference_mode()
def _get_embs_fn(loader, model, device):
    results = []
    model.eval()
    model.to(device)
    tk0 = tqdm(loader, total=len(loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        embs = model(inputs)
        results.append(embs.to("cpu").numpy())
    results = np.concatenate(results)
    return results


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transformer_embs(
    texts: list[list[str]],
    model_name: str,
    tokenizer_class_name: str | None = None,
    batch_size=32,
    max_len=512,
    device=_DEVICE,
    pooling="mean",
):
    embs = []
    model = TransformerEmbsModel(model_name, pooling)
    if tokenizer_class_name is None:
        tokenizer_class = AutoTokenizer
    else:
        tokenizer_class = getattr(transformers, tokenizer_class_name)
    tokenizer = tokenizer_class.from_pretrained(model_name)
    for text in texts:
        dataset = TextDataset(text, tokenizer, max_len)
        # for d in tqdm(dataset, total=len(dataset)):
        #     pass
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        emb = _get_embs_fn(loader, model, device)
        embs.append(emb)
    del model
    return embs


if __name__ == "__main__":
    texts = [
        ["こんにちは 世界", "こんばんわ 世界"],
        ["hello world", "goodnight world"],
    ]
    embs = get_transformer_embs(texts, "bert-base-uncased")
    print(np.array(embs).shape, embs[0][0:5])
    embs = get_transformer_embs(texts, "bert-base-uncased", pooling="cls")
    print(np.array(embs).shape, embs[0][0:5])
