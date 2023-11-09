import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DefaultDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str,
        max_length: int = 256,
        return_labels: bool = True,
    ):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.return_labels = return_labels

        self.texts = df["cleansed_text"].tolist()
        self.labels = df["generated"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        inputs = self.prepare_input(
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            text=self.texts[idx],
        )
        if self.return_labels:
            labels = torch.Tensor([self.labels[idx]])
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels,
            }

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    @staticmethod
    def prepare_input(max_length: int, tokenizer: AutoTokenizer, text: str) -> dict:
        inputs = tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs
