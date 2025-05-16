import json
import torch
from torch.utils.data import Dataset

class PerguntasRespostasDataset(Dataset):
    def __init__(self, caminho_arquivo, tokenizer, max_length=512):
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            self.pares = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        item = self.pares[idx]
        pergunta = item['pergunta']
        resposta = item['resposta']

        encoding = self.tokenizer(
            pergunta,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            resposta,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }
