import os
import json
from glob import glob
import shutil
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt

from config import BASE_DIR, RESULTADOS_DIR, DATASET_COM_TESTE


def avaliar_e_predizer(modelo, nome_subpasta="predicoes", salvar_grafico=True):
    """
    Avalia o modelo no conjunto de teste (quantitativo e visual).
    - Salva métricas em JSON e gráfico PNG.
    - Salva imagens com bounding boxes desenhadas.
    """

    # === Avaliação quantitativa ===
    print("\n🔎 Avaliando modelo no conjunto de teste...")

    avaliacao = modelo.val(
        data=os.path.join(BASE_DIR, "data.yaml"),
        split="test",
        project="runs",     # muda o diretório base para "runs/test"
        name="teste",          # subpasta (pode ser automático tipo datetime)
        save=True,               # salva imagens com bounding boxes
        save_txt=True,           # salva labels preditas em txt (formato YOLO)
        verbose=False
    )


    # Salvar métricas
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    path_json = os.path.join(RESULTADOS_DIR, "metricas_teste.json")
    with open(path_json, "w") as f:
        json.dump(avaliacao.results_dict, f, indent=4)

    print("\n📈 Métricas de avaliação no conjunto de teste:")
    for chave, valor in avaliacao.results_dict.items():
        print(f"{chave}: {valor}")
