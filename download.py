import os
from roboflow import Roboflow
from config import API_KEY, LISTA_DATASETS, BASE_DIR, RESULTADOS_DIR
from logger import logger

def baixar_datasets() -> None:
    """
    Baixa os datasets especificados na configuração a partir da plataforma Roboflow.

    Os datasets são baixados e salvos na pasta definida por BASE_DIR.
    Evita downloads duplicados caso os dados já estejam presentes localmente.
    Os logs das operações são registrados usando o logger centralizado.

    Parâmetros
    ----------
    Nenhum

    Retorno
    -------
    None
        Esta função não retorna nenhum valor. Apenas realiza o download, salva os dados e registra logs.
    """
    
    # Garante que as pastas existam
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    roboflow = Roboflow(api_key=API_KEY)

    for workspace, projeto, versao in LISTA_DATASETS:
        nome_pasta = f"{projeto}_v{versao}"
        destino = os.path.join(BASE_DIR, nome_pasta)

        if os.path.exists(destino):
            logger.warning(f"Dataset já existente: {destino}")
            continue

        logger.info(f"Baixando: {workspace}/{projeto} (versão {versao})")
        projeto_rf = roboflow.workspace(workspace).project(projeto)
        projeto_rf.version(versao).download("yolo11", location=destino)
        logger.info(f"Download concluído: {destino}")
