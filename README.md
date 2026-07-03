# 🐆 Kurupyra Cortes

Pipeline de automação em Python para geração de cortes virais a partir de conteúdo de podcast, com integração à API do Gemini para análise e seleção de trechos.

> **Nota sobre uso de IA:** grande parte do código-fonte deste projeto foi desenvolvida com apoio de ferramentas de IA generativa. Meu foco pessoal esteve na arquitetura do pipeline, na integração entre os módulos e no diagnóstico e resolução de problemas reais (depreciação de SDK, esgotamento de quota de API, otimização de chamadas).

## 📋 Sobre o projeto

O Kurupyra Cortes automatiza o processo de transformar um episódio de podcast bruto em cortes curtos prontos para publicação (formato vertical, com legendas), reduzindo o trabalho manual de edição repetitivo.

## ⚙️ Como funciona (pipeline)

O processamento é dividido em módulos sequenciais:

1. **`modulo1_transcricao.py`** — Transcreve o áudio do podcast bruto, com suporte a aceleração por GPU (CUDA) para reduzir o tempo de processamento
2. **`modulo2_diretor.py`** — Envia a transcrição para a API do Gemini, que analisa o conteúdo e identifica os trechos com maior potencial viral
3. **`modulo3_corte.py`** — Corta os trechos selecionados do vídeo original e ajusta o formato para vertical (short-form)
4. **`modulo4_legendas.py`** — Gera e insere legendas automáticas nos cortes finais
5. **`main.py`** — Orquestra a execução completa do pipeline, do vídeo bruto ao corte pronto para postagem

## 🧠 Desafios técnicos resolvidos

- **Falhas em cascata por SDK depreciada:** o pipeline parou de funcionar após a descontinuação do SDK e do modelo do Gemini utilizados originalmente; diagnostiquei a causa raiz e migrei a integração para a versão atual da API
- **Esgotamento de quota na API gratuita:** identifiquei que o alto número de chamadas por vídeo processado esgotava a cota do plano gratuito
- **Em andamento:** implementação de *batching* de chamadas à API do Gemini, para reduzir o número de requisições por vídeo processado

## 🛠️ Tecnologias

- **Python** — linguagem principal do pipeline
- **API do Gemini** — seleção de trechos e processamento de conteúdo
- **CUDA** — aceleração de GPU para transcrição (ver `INSTRUCOES_CUDA_WINDOWS.md`)
- **PowerShell** — script de instalação de ambiente (`install_windows_cuda.ps1`)

## 🚀 Como rodar

```bash
# Clone o repositório
git clone https://github.com/rafaelqueiroxz/kurupyra_cortes.git
cd kurupyra_cortes

# Instale as dependências
pip install -r requirements.txt

# (Opcional, Windows) Configure aceleração CUDA
# Veja instruções detalhadas em INSTRUCOES_CUDA_WINDOWS.md

# Configure sua chave de API do Gemini
# Crie um arquivo .env na raiz do projeto com:
# GEMINI_API_KEY=sua_chave_aqui

# Execute o pipeline
python main.py
```

## 📁 Estrutura de pastas

```
kurupyra_cortes/
├── main.py                      # Orquestrador do pipeline
├── modulo1_transcricao.py       # Transcrição de áudio
├── modulo2_diretor.py           # Seleção de trechos via API Gemini
├── modulo3_corte.py             # Corte e reformatação de vídeo
├── modulo4_legendas.py          # Geração de legendas
├── requirements.txt             # Dependências do projeto
├── podcasts_brutos/             # Vídeos de entrada (não versionado)
├── cortes_prontos/              # Cortes finais (não versionado)
└── assets/                      # Recursos auxiliares
```

> Pastas de vídeo (`podcasts_brutos/`, `cortes_prontos/`, `cortes_temp/`, `output/`) não são versionadas neste repositório — contêm apenas mídia bruta/processada, não código-fonte.

## 👤 Autor

**Rafael Queiroz**
Estudante de Análise e Desenvolvimento de Sistemas
[GitHub](https://github.com/rafaelqueiroxz) · [LinkedIn](https://www.linkedin.com/in/rafael-queiroz-8860073b6/)
