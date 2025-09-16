import os
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt

from config import BASE_DIR, RESULTADOS_DIR, DATASET_COM_TESTE
from logger import logger


def avaliar_e_predizer(modelo, nome_subpasta="predicoes") -> None:
    """
    Avalia o modelo no conjunto de teste de forma quantitativa e visual.
    Salva métricas em arquivo JSON, imagens com bounding boxes e, opcionalmente, gráfico PNG.

    Parâmetros
    ----------
    modelo : YOLO
        Modelo YOLO previamente treinado.
    nome_subpasta : str, opcional
        Nome da subpasta dentro de RESULTADOS_DIR onde salvar as predições (padrão: 'predicoes').
    salvar_grafico : bool, opcional
        Se True, gera e salva gráfico das métricas (padrão: True).

    Retorno
    -------
    None
        Função realiza salvamento dos resultados e não retorna valores.
    """

    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    logger.info("Iniciando avaliação do modelo no conjunto de teste.")

    avaliacao = modelo.val(
        data=os.path.join(BASE_DIR, "data.yaml"),
        split="test",
        project=RESULTADOS_DIR,
        name=nome_subpasta,
        save=True,          # salva imagens com bounding boxes
        save_txt=True,      # salva labels preditas em txt (formato YOLO)
        verbose=False
    )

    # Salvar métricas em JSON
    path_json = os.path.join(RESULTADOS_DIR, "metricas_teste.json")
    with open(path_json, "w") as f:
        json.dump(avaliacao.results_dict, f, indent=4)

    logger.info(f"Métricas salvas em: {path_json}")
    logger.info("Métricas de avaliação no conjunto de teste:")
    for chave, valor in avaliacao.results_dict.items():
        logger.info(f"{chave}: {valor}")

    logger.info("Avaliação concluída.")