import random
import json
from pathlib import Path

equipamentos = ["notebook", "PC", "roteador", "monitor", "HD externo", "impressora"]
problemas = ["não liga", "está lento", "fica travando", "não conecta no Wi-Fi", "mostra tela azul", "faz barulho estranho"]
erros = ["erro 0x80070005", "erro de driver", "erro de DLL", "sistema não encontrado", "acesso negado"]
sistemas = ["Windows 10", "Windows 11", "Ubuntu", "Linux Mint"]
programas = ["Google Chrome", "Photoshop", "Office", "Steam", "Antivírus"]
respostas = [
    "Tente reiniciar o equipamento e verificar as conexões.",
    "Atualize os drivers pelo Gerenciador de Dispositivos.",
    "Execute uma verificação com o antivírus.",
    "Restaure o sistema para um ponto anterior.",
    "Verifique se há atualizações disponíveis para o sistema.",
    "Desinstale e reinstale o programa afetado."
]

def gerar_par(perguntas, respostas, quantidade=500):
    pares = []
    for _ in range(quantidade):
        tipo = random.choice(["equipamento", "erro", "programa"])
        if tipo == "equipamento":
            pergunta = f"Meu {random.choice(equipamentos)} {random.choice(problemas)}, o que posso fazer?"
        elif tipo == "erro":
            pergunta = f"Como resolver o {random.choice(erros)} no {random.choice(sistemas)}?"
        else:
            pergunta = f"O {random.choice(programas)} está travando no {random.choice(sistemas)}. Alguma solução?"
        resposta = random.choice(respostas)
        pares.append({"pergunta": pergunta, "resposta": resposta})
    return pares

pares_gerados = gerar_par([], respostas, quantidade=500)

Path("dados/brutos/processados/perguntas_e_respostas").mkdir(parents=True, exist_ok=True)
with open("dados/brutos/processados/perguntas_e_respostas/perguntas_respostas.json", "w", encoding="utf-8") as f:
    json.dump(pares_gerados, f, indent=2, ensure_ascii=False)