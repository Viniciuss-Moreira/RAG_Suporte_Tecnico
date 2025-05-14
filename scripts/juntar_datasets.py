import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_DADOS = os.path.join(BASE_DIR, "../dados/brutos/processados")
ARQUIVO_SAIDA = os.path.join(BASE_DIR, "../dados/brutos/processados/todos_dados_consolidados.csv")

arquivos = [
    "IT_tickets_kaggle.csv",
    "qa_superuser.csv",
    "salarios_de_tecnologia.csv",
    "serverfault.csv",
    "twitterTickets.csv",
]

dataframes = []
for nome_arquivo in arquivos:
    caminho_completo = os.path.join(CAMINHO_DADOS, nome_arquivo)
    try:
        df = pd.read_csv(caminho_completo, encoding="utf-8")
        dataframes.append(df)
        print(f"ok {nome_arquivo}")
    except Exception as e:
        print(f"erro {nome_arquivo}: {e}")

if dataframes:
    df_concatenado = pd.concat(dataframes, ignore_index=True)
    df_concatenado.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")
    print(f"ok {ARQUIVO_SAIDA}")
else:
    print("erro")
