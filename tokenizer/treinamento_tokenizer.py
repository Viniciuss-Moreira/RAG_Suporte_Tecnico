from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# Criação e treino
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
trainer = BpeTrainer(special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
tokenizer.train(["../dados/brutos/processados/corpus.txt"], trainer)

# Salvar vocabulário e merges
tokenizer.save("tokenizer_treinado/tokenizer.json")

# Converter para Hugging Face e salvar
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_treinado/tokenizer.json",
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>"
)
hf_tokenizer.save_pretrained("tokenizer_treinado")
