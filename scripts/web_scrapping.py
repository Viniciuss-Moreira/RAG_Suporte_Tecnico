import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_SAIDA = os.path.join(BASE_DIR, "../dados/brutos/documentacao.txt")

os.makedirs(os.path.dirname(ARQUIVO_SAIDA), exist_ok=True)

def coletar_com_paginacao(url_bases, num_paginas):
    for url_base in url_bases:
        for i in range(1, num_paginas + 1):
            url = f"{url_base}{i}"
            try:
                resposta = requests.get(url, timeout=10)
                resposta.raise_for_status()
                soup = BeautifulSoup(resposta.text, "html.parser")

                paragrafos = soup.find_all("p")
                with open(ARQUIVO_SAIDA, "a", encoding="utf-8") as f:
                    for p in paragrafos:
                        texto = p.get_text(strip=True)
                        if texto:
                            f.write(texto + "\n\n")
                print(f"ok {url}")
            except Exception as e:
                print(f"erro {url}: {e}")

def coletar_links_internos(url_bases):
    for url_base in url_bases:
        try:
            resposta = requests.get(url_base, timeout=10)
            resposta.raise_for_status()
            soup = BeautifulSoup(resposta.text, "html.parser")

            links = soup.find_all("a", href=True)
            urls = [urljoin(url_base, link["href"]) for link in links if link["href"].startswith("/docs")]

            for url in urls:
                try:
                    resposta = requests.get(url, timeout=10)
                    resposta.raise_for_status()
                    soup = BeautifulSoup(resposta.text, "html.parser")

                    paragrafos = soup.find_all("p")
                    with open(ARQUIVO_SAIDA, "a", encoding="utf-8") as f:
                        for p in paragrafos:
                            texto = p.get_text(strip=True)
                            if texto:
                                f.write(texto + "\n\n")
                    print(f"ok {url}")
                except Exception as e:
                    print(f"erro {url}: {e}")
        except Exception as e:
            print(f"erro {url_base}")

def coletar_de_urls(urls):
    with open(ARQUIVO_SAIDA, "a", encoding="utf-8") as f:
        for url in urls:
            try:
                resposta = requests.get(url, timeout=10)
                resposta.raise_for_status()
                soup = BeautifulSoup(resposta.text, "html.parser")

                paragrafos = soup.find_all("p")
                for p in paragrafos:
                    texto = p.get_text(strip=True)
                    if texto:
                        f.write(texto + "\n\n")
                print(f"[✔] Conteúdo extraído de: {url}")
            except Exception as e:
                print(f"[✘] Erro ao processar {url}: {e}")

def main():
    urls = [
        "https://huggingface.co/docs/transformers/index",
        "https://huggingface.co/docs/transformers/installation",
        "https://huggingface.co/docs/transformers/model_doc/gpt2",
        "https://github.com/huggingface/transformers/issues",
        "https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter/data",
        "https://archive.org/details/stackexchange",
        "",
    ]
    
    print("scaping em andamento em todas as paginas fornecidas")
    coletar_de_urls(urls)

    print("paginando as paginas fornecidas")
    url_bases_paginacao = [
        "https://huggingface.co/docs/transformers/page/",
        "https://example.com/docs/page/",
    ]
    num_paginas = 5
    coletar_com_paginacao(url_bases_paginacao, num_paginas)

    print("coletando todos os links internos das paginas")
    url_bases_links = [
        "https://huggingface.co/docs/transformers/index",
        "https://huggingface.co/docs/transformers/installation",
        "https://huggingface.co/docs/transformers/model_doc/gpt2"
    ]
    coletar_links_internos(url_bases_links)

    print("coleta concluida")

if __name__ == "__main__":
    main()
