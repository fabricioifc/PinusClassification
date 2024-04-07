# Pinus ou Não Pinus - Classificação de Imagens

## Descrição do Projeto

O projeto tem como objetivo classificar imagens de pinus e não pinus. As imagens aéreas foram coletadas por um drone e são utilizadas para identificar a presença de pinus em uma determinada área. O modelo de classificação foi treinado com imagens de pinus e não pinus.

## Dataset

O dataset utilizado para treinar o modelo foi criado a partir de imagens aéreas coletadas por um drone. As imagens foram extraidas de um vídeo e foram classificadas manualmente. O dataset está dividido em três pastas: treino, validação e teste. Cada pasta contém duas subpastas: pinus e nao_pinus. As imagens estão no formato png na seguinte estrutura de diretórios:

```bash
data
│
└───train
│   └───pinus `(533 images)`
│   └───notPinus `(290 images)`
│
└───val
│   └───pinus `(150 images)`
│   └───notPinus `(56 images)`
│
└───test
    └───pinus `(149 images)`
    └───notPinus `(56 images)`
```

## Como executar o projeto

Antes de executar o projeto, é necessário instalar o Python e o pip. O projeto foi desenvolvido com Python `3.10.11`. Em seguida, crie um ambiente virtual para instalar as bibliotecas necessárias. Para criar o ambiente virtual, execute o comando abaixo:

```bash
python -m venv .venv
```

Em seguida, ative o ambiente virtual. No Windows, execute o comando abaixo:

```bash
.venv\Scripts\activate
```

## Instalando as bibliotecas

Para executar o projeto, é necessário instalar as bibliotecas necessárias. O arquivo `requirements.txt` contém todas as bibliotecas necessárias para executar o projeto. Para instalar as bibliotecas, execute o comando abaixo:

```bash
pip install -r requirements.txt
```

## Instalando o PyTorch

O projeto utiliza a biblioteca PyTorch para treinar o modelo de classificação. Para instalar o PyTorch, acesse o site oficial da biblioteca e siga as instruções de instalação para o seu sistema operacional. O PyTorch pode ser instalado com ou sem suporte para GPU. Para instalar o PyTorch com suporte para GPU, é necessário instalar o CUDA Toolkit. Para instalar o PyTorch sem suporte para GPU, execute o comando abaixo:

Mais detalhes sobre a instalação do PyTorch podem ser encontrados no site oficial da biblioteca: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Treinamento do Modelo

Após instalar as bibliotecas, execute o arquivo main.py para treinar o modelo e classificar as imagens. Use o comando abaixo:

```bash
python main.py --epochs=10 --batch_size=4
```

onde:
- *--epochs*: número de épocas para treinar o modelo (opcional)
- *--batch_size*: tamanho do batch para treinar o modelo (opcional)

## Treinamento do Modelo em Background

Para treinar o modelo em background, execute o comando abaixo:

```bash
nohup python main.py --epochs=10 --batch_size=4 > output.log &
```

 - *Observação*: o comando acima só funciona em sistemas Linux/Debian. Instale o pacote `coreutils` para utilizar o comando `nohup`. Para instalar o pacote, execute o comando abaixo:

```bash
sudo apt-get install coreutils
```

## Apenas Teste

Para testar o modelo salvo, execute o arquivo main.py com a opção `--test`:

```bash
python main.py --test
```

## Resultados

Os resultados serão salvos na pasta `results` e serão exibidos no console.