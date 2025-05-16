import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AdamW, get_scheduler, AutoTokenizer
from dataset import PerguntasRespostasDataset
from tqdm import tqdm

def main():
    # Configurações
    caminho_dataset = "dados/brutos/processados/perguntas_e_respostas/perguntas_respostas.json"  # caminho do seu JSON com perguntas e respostas
    modelo_nome = "t5-small"
    batch_size = 8
    epochs = 3
    learning_rate = 5e-5
    max_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    model = AutoModelForSeq2SeqLM.from_pretrained(modelo_nome)
    model.to(device)

    dataset = PerguntasRespostasDataset(caminho_dataset, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.train()

    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    model.save_pretrained("modelo/treinado")
    tokenizer.save_pretrained("modelo/treinado")

if __name__ == "__main__":
    main()
