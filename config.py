"""
Módulo de configuração para o pipeline de treinamento e avaliação com YOLO.
Define constantes utilizadas ao longo do projeto, incluindo caminhos, API key,
modelos e especificações dos datasets.
"""

import os

# API Roboflow
API_KEY = 'API_KEY para o Dataset'

# Modelo YOLO
YOLO_MODELO = "yolo11n"
NOME_MODELO = f"{YOLO_MODELO}.pt"

# Diretórios principais
BASE_DIR = os.path.join(os.getcwd(), "datasets")
RESULTADOS_DIR = os.path.join(os.getcwd(), "resultados")

# Dataset que contém um conjunto de teste
DATASET_COM_TESTE = "itens-de-dispensa-8pudf_v4"

# Lista de datasets adicionais no formato (workspace, projeto, versão)
LISTA_DATASETS = [
    ("identvintern", "groceries-9vwuo", 3),
    ("ic-rfkuy", "itens-de-dispensa-8pudf", 4),
]
