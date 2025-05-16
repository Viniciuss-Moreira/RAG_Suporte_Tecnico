import yaml
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import os
from treinamento.dataset import PerguntasRespostasDataset

def train():
    with open("treinamento/configuracao.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name_or_path"])
    model.to(device)

    dataset = PerguntasRespostasDataset(config["dataset_path"], tokenizer, max_length=config["max_length"])
    val_size = int(len(dataset) * config.get("val_split", 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    num_training_steps = config["num_train_epochs"] * len(train_loader)
    lr_scheduler = get_scheduler(
        name=config.get("lr_scheduler", "linear"),
        optimizer=optimizer,
        num_warmup_steps=config.get("warmup_steps", 0),
        num_training_steps=num_training_steps
    )

    best_val_loss = float("inf")

    for epoch in range(config["num_train_epochs"]):
        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            labels = labels.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_train_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                labels = labels.clone()
                labels[labels == tokenizer.pad_token_id] = -100

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        output_dir = config["output_dir"]
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Checkpoint salvo no epoch {epoch+1} com val loss {avg_val_loss:.4f}")

if __name__ == "__main__":
    train()
