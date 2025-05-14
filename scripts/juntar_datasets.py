import pandas as pd
import os

caminho_arquivos = '/dados/brutos/processados/'

arquivos_csv = [
    'IT_Tickets_kaggle.csv',
    'qa_superuser.csv',
    'salarios_de_tecnologia.csv',
    'serverfault.csv',
    'twitterTickets.csv'
]

dataframes = []

colunas_unicas = set()

for arquivo in arquivos_csv:
    caminho_arquivo = os.path.join(caminho_arquivos, arquivo)
    df = pd.read_csv(caminho_arquivo)
    
    colunas_unicas.update(df.columns)
    
    dataframes.append(df)

for i, df in enumerate(dataframes):
    for coluna in colunas_unicas:
        if coluna not in df.columns:
            df[coluna] = pd.NA
    
    dataframes[i] = df[sorted(colunas_unicas)]

df_final = pd.concat(dataframes, ignore_index=True)

output_file = '/dados/brutos/processados/corpus_unificado.csv'
df_final.to_csv(output_file, index=False)

print(f'Dataset unificado e ajustado foi salvo em: {output_file}')
