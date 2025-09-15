import os
import json
from ultralytics import YOLO
import torch
import torch.nn as nn
from config import BASE_DIR, RESULTADOS_DIR

def treinar_modelo(modelo):
    """
    Treina o modelo YOLO, descongelando as 5 últimas camadas, salva pesos e arquitetura,
    e retorna o modelo treinado.
    """
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    # 🔓 Descongelar as últimas 5 camadas
    '''parametros = list(modelo.model.named_parameters())
    total_camadas = len(parametros)
    for i, (nome, parametro) in enumerate(parametros):
        parametro.requires_grad = (i >= total_camadas - 5)

    print("\n🔍 Camadas descongeladas:")
    for nome, parametro in parametros[-5:]:
        print(f"{nome} | requires_grad = {parametro.requires_grad}")'''
    
    # Obter todas as camadas reais do modelo
    camadas = list(modelo.model.modules())

    # Filtrar apenas camadas com parâmetros treináveis (e evitar duplicadas)
    camadas_com_parametros = [c for c in camadas if any(p.requires_grad is not None for p in c.parameters())]

    # Congelar todas as camadas
    '''for camada in camadas_com_parametros:
        for p in camada.parameters():
            p.requires_grad = False'''

    # Descongelar apenas as 5 últimas camadas com parâmetros
    for camada in reversed(camadas_com_parametros[-5:]):
        for p in camada.parameters():
            p.requires_grad = True

    # Verificação: imprime as camadas que foram descongeladas
    #print("\n🔓 Últimas 5 camadas descongeladas:")
    #for i, camada in enumerate(reversed(camadas_com_parametros[-5:]), 1):
    #    print(f"{i}: {type(camada).__name__}")

    # Verificação de contagem
    total = sum(p.numel() for p in modelo.model.parameters())
    treinaveis = sum(p.numel() for p in modelo.model.parameters() if p.requires_grad)
    print(f"\n🧠 Total de parâmetros: {total:,}")
    print(f"🎯 Parâmetros treináveis: {treinaveis:,}")

    # 🚀 Treinamento
    modelo.train(
        data=os.path.join(BASE_DIR, "data.yaml"),
        epochs=15, # Reduzido para 1 para testes rápidos
        lr0=0.001,
        batch=16,
        imgsz=640,
        patience=3,
        verbose=False
    )

    # 🧠 Parâmetros do modelo
    total = sum(p.numel() for p in modelo.model.parameters())
    treinaveis = sum(p.numel() for p in modelo.model.parameters() if p.requires_grad)
    print(f"\n🧠 Total de parâmetros: {total:,}")
    print(f"🎯 Parâmetros treináveis: {treinaveis:,}")

    # 💾 Salvando o modelo
    modelo.save(os.path.join(RESULTADOS_DIR, "modelo_treinado.pt"))
    print(f"✅ Modelo salvo em: {RESULTADOS_DIR}")

    # 📝 Salvando a arquitetura
    with open(os.path.join(RESULTADOS_DIR, "arquitetura.txt"), "w") as f:
        f.write(str(modelo.model))

    # 📊 Avaliação opcional imediata
    aval = modelo.val(data=os.path.join(BASE_DIR, "data.yaml"), split="test")
    with open(os.path.join(RESULTADOS_DIR, "metricas.json"), "w") as f:
        json.dump(aval.results_dict, f, indent=4)

    print(f"📊 Métricas e arquitetura salvas na pasta: {RESULTADOS_DIR}")

    return modelo  # ✅ Retorna modelo treinado
