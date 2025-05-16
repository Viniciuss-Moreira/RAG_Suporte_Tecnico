import json
from torch.utils.data import Dataset

class PerguntasRespostasDataset(Dataset):
    def __init__(self, caminho_arquivo, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            self.dados = json.load(f)

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx):
        item = self.dados[idx]
        pergunta = item["pergunta"]
        resposta = item["resposta"]

        inputs = self.tokenizer(
            pergunta,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer(
            resposta,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = targets["input_ids"].squeeze()

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
