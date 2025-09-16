import os
import json
import random
import numpy as np
import torch
from config import BASE_DIR, RESULTADOS_DIR
from logger import logger  # Supondo que você tenha um módulo logger.py

def setar_seed(seed: int = 42) -> None:
    """
    Define a seed para garantir reprodutibilidade do treinamento.

    Parâmetros
    ----------
    seed : int, opcional
        Valor da seed (padrão: 42).

    Retorno
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def treinar_modelo(modelo) -> object:
    """
    Realiza o treinamento do modelo YOLO, descongelando as últimas 5 camadas,
    salva os pesos e a arquitetura do modelo, além de avaliar e salvar métricas.

    Parâmetros
    ----------
    modelo : ultralytics.YOLO
        Modelo YOLO carregado e pronto para treino.

    Retorno
    -------
    modelo : ultralytics.YOLO
        Modelo treinado com os pesos atualizados.
    """
    setar_seed(42)
    
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    # Descongelar as últimas 5 camadas com parâmetros treináveis
    camadas = list(modelo.model.modules())
    camadas_com_parametros = [c for c in camadas if any(p.requires_grad is not None for p in c.parameters())]

    for camada in reversed(camadas_com_parametros[-5:]):
        for p in camada.parameters():
            p.requires_grad = True

    total = sum(p.numel() for p in modelo.model.parameters())
    treinaveis = sum(p.numel() for p in modelo.model.parameters() if p.requires_grad)
    logger.info(f"Total de parâmetros: {total:,}")
    logger.info(f"Parâmetros treináveis: {treinaveis:,}")

    # Treinamento do modelo
    modelo.train(
        data=os.path.join(BASE_DIR, "data.yaml"),
        epochs=15,
        lr0=0.001,
        batch=16,
        imgsz=640,
        patience=3,
        verbose=False
    )

    total = sum(p.numel() for p in modelo.model.parameters())
    treinaveis = sum(p.numel() for p in modelo.model.parameters() if p.requires_grad)
    logger.info(f"Após treino - Total de parâmetros: {total:,}")
    logger.info(f"Após treino - Parâmetros treináveis: {treinaveis:,}")

    # Salvar modelo e arquitetura
    modelo_path = os.path.join(RESULTADOS_DIR, "modelo_treinado.pt")
    modelo.save(modelo_path)
    logger.info(f"Modelo salvo em: {modelo_path}")

    arquitetura_path = os.path.join(RESULTADOS_DIR, "arquitetura.txt")
    with open(arquitetura_path, "w") as f:
        f.write(str(modelo.model))

    # Avaliação pós-treinamento
    aval = modelo.val(data=os.path.join(BASE_DIR, "data.yaml"), split="test")
    metricas_path = os.path.join(RESULTADOS_DIR, "metricas.json")
    with open(metricas_path, "w") as f:
        json.dump(aval.results_dict, f, indent=4)

    logger.info(f"Métricas e arquitetura salvas em: {RESULTADOS_DIR}")

    return modelo
