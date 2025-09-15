import os
from roboflow import Roboflow
from config import API_KEY, LISTA_DATASETS, BASE_DIR

def baixar_datasets():
    os.makedirs(BASE_DIR, exist_ok=True)
    roboflow = Roboflow(api_key=API_KEY)

    for workspace, projeto, versao in LISTA_DATASETS:
        nome_pasta = f"{projeto}_v{versao}"
        destino = os.path.join(BASE_DIR, nome_pasta)

        if os.path.exists(destino):
            print(f"⚠️ Já existe: {destino}")
            continue

        print(f"🔽 Baixando: {workspace}/{projeto} v{versao}")
        projeto_rf = roboflow.workspace(workspace).project(projeto)
        projeto_rf.version(versao).download("yolo11", location=destino)
        print(f"✅ Salvo em: {destino}")
