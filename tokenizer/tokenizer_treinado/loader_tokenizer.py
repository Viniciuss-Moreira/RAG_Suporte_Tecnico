from transformers import PreTrainedTokenizerFast
import json

tokenizer = PreTrainedTokenizerFast.from_pretrained("../tokenizer_treinado", local_files_only=True)

with open("../../dados/brutos/processados/corpus.txt", "r", encoding="utf-8") as f:
    linhas = f.readlines()

tokenized_texts = [tokenizer(text.strip(), truncation=True, padding="max_length", max_length=256).data for text in linhas]

with open("dados_tokenizados.json", "w", encoding="utf-8") as f:
    json.dump(tokenized_texts, f, ensure_ascii=False, indent=2)
