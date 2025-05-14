from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing 

corpus_file = "../dados/brutos/processados/corpus.txt"
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[corpus_file],
    vocab_size=30522,
    min_frequency=2,
    show_progress=True,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]
)

output_dir = "./meu_tokenizer_custom"
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tokenizer.save_model(output_dir)

print(f"tokenizer treinado em {output_dir}")