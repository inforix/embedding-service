# Embedding as Service

provides a service for embedding

install miniconda

## Create environment

```bash
conda create -n embeddingservice
conda activate embeddingservice
```

## Install packages

```bash
pip install pip -U
pip install -r requirements.txt
```

## Create .env file

```bash
cp .env.example .env
```

## Run

```bash
uvicon main:app --reload --host 0.0.0.0
```
