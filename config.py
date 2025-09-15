import os

# 🔑 API Roboflow
API_KEY = "ZWwm2v5dktzS4yL27POm"

# 🔍 Modelo YOLO
YOLO_MODELO = "yolo11n"
NOME_MODELO = f"{YOLO_MODELO}.pt"

# 📁 Diretórios
BASE_DIR = os.path.join(os.getcwd(), "datasets")
RESULTADOS_DIR = os.path.join(os.getcwd(), "resultados")

# 🧪 Dataset com conjunto de teste
DATASET_COM_TESTE = "itens-de-dispensa-8pudf_v4"

# 📦 Lista de datasets (workspace, projeto, versão)
LISTA_DATASETS = [
    ("identvintern", "groceries-9vwuo", 3),
    ("ic-rfkuy", "itens-de-dispensa-8pudf", 4),
]
