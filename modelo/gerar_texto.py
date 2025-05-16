from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("/workspaces/RAG_Suporte_Tecnico/modelo/treinado", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("/workspaces/RAG_Suporte_Tecnico/modelo/treinado", local_files_only=True)

entrada = "pergunta: Meu notebook faz barulho estranho, o que posso fazer?"
inputs = tokenizer(entrada, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_length=100,
    num_beams=4,
    early_stopping=True
)

resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Resposta:", resposta)
