import os
import yaml
from config import BASE_DIR, DATASET_COM_TESTE

def gerar_data_yaml():
    subpastas = [p for p in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, p))]
    lista_classes = []
    caminhos_treino, caminhos_valid, caminhos_teste = [], [], []

    for pasta in subpastas:
        yaml_path = os.path.join(BASE_DIR, pasta, "data.yaml")
        if not os.path.exists(yaml_path):
            continue

        with open(yaml_path, "r") as f:
            dados = yaml.safe_load(f)
            nomes = dados.get("names")
            if isinstance(nomes, list):
                lista_classes.append(set(nomes))

        caminho_treino = os.path.join(pasta, "train", "images")
        caminho_valid = os.path.join(pasta, "valid", "images")

        if os.path.exists(os.path.join(BASE_DIR, caminho_treino)):
            caminhos_treino.append(caminho_treino)
        if os.path.exists(os.path.join(BASE_DIR, caminho_valid)):
            caminhos_valid.append(caminho_valid)

        if pasta == DATASET_COM_TESTE:
            caminho_teste = os.path.join(pasta, "test", "images")
            if os.path.exists(os.path.join(BASE_DIR, caminho_teste)):
                caminhos_teste.append(caminho_teste)

    if not lista_classes:
        raise RuntimeError("❌ Nenhum dataset com classes válidas encontrado.")

    classes_comuns = sorted(set.intersection(*lista_classes)) if len(lista_classes) > 1 else sorted(lista_classes[0])

    yaml_final = {
        "path": BASE_DIR,
        "train": caminhos_treino,
        "val": caminhos_valid,
        "test": caminhos_teste,
        "nc": len(classes_comuns),
        "names": classes_comuns
    }

    caminho_yaml = os.path.join(BASE_DIR, "data.yaml")
    with open(caminho_yaml, "w") as f:
        yaml.safe_dump(yaml_final, f)

    print(f"✅ Arquivo data.yaml criado em: {caminho_yaml}")
    print("📁 Conjuntos incluídos:")
    print("• Treino:", caminhos_treino)
    print("• Validação:", caminhos_valid)
    print("• Teste:", caminhos_teste)
    print("• Classes:", classes_comuns)
