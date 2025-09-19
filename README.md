
# Inventario de Alimentos

Sistema completo para inventário automático de alimentos usando detecção de objetos com YOLO, integração com datasets do Roboflow e interface gráfica em KivyMD.

## Sumário
- [Descrição](#descrição)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Dependências](#dependências)
- [Como Usar](#como-usar)
- [Datasets](#datasets)
- [Treinamento e Avaliação](#treinamento-e-avaliação)
- [Resultados](#resultados)
- [Interface Gráfica](#interface-gráfica)
- [Autores](#autores)

---

## Descrição
Este projeto realiza o inventário automático de itens alimentícios a partir de imagens, utilizando modelos YOLO treinados com dados de múltiplos datasets. O pipeline inclui download automático dos dados, preparação, treinamento, avaliação, geração de métricas e interface gráfica para visualização e ajuste dos resultados.

## Estrutura do Projeto

```
inventario_de_alimentos/
│
├── app.py              # Interface gráfica KivyMD para visualização e ajuste do inventário
├── config.py           # Configurações globais, caminhos, API keys, nomes de datasets
├── download.py         # Download automatizado dos datasets do Roboflow
├── prepare_data.py     # Geração do arquivo data.yaml consolidado
├── train.py            # Treinamento do modelo YOLO
├── predict.py          # Avaliação e predição no conjunto de teste
├── logger.py           # Configuração centralizada de logging
├── main.py             # Pipeline completo: download, preparação, treino, avaliação
├── requirements.txt    # Dependências do projeto
├── datasets/           # Datasets baixados e arquivo data.yaml consolidado
├── resultados/         # Pesos, métricas, logs, predições e gráficos
└── runs/               # Resultados de execuções/testes
```

## Apresentação sobre o projeto

[Apresentação em PDF](https://drive.google.com/file/d/1bHjLoTAZN6plcOSWAXALusHv1ps2Cmp_/view?usp=sharing)

## Clone o repositório

```bash
git clone https://github.com/elencris/inventario_de_alimentos.git
cd inventario_de_alimentos
```

## Crie e ative o ambiente virtual

```bash
python -m venv ml-env

# Linux
source ml-env/bin/activate

# Windows
.\ml-env\Scripts\activate
```

## Dependências
Instale as dependências com:

```bash
pip install -r requirements.txt
```

Principais pacotes:
- ultralytics (YOLO)
- roboflow
- pyyaml
- matplotlib
- kivymd, kivy (para interface gráfica)

## Como Usar

1. **Configuração:**
	- Edite `config.py` para ajustar caminhos, API key do Roboflow e nomes dos datasets.

2. **Executar pipeline completo:**
	- No terminal:
	  ```bash
	  python main.py
	  ```
	- Isso irá baixar os datasets, gerar o `data.yaml`, treinar o modelo e avaliar no conjunto de teste.

3. **Interface gráfica:**
	- Após o treinamento, execute:
	  ```bash
	  python app.py
	  ```
	- A interface permite visualizar e ajustar o inventário detectado a partir de imagens.

## Datasets

Os datasets são baixados automaticamente do Roboflow, conforme especificado em `config.py`:

- [groceries-9vwuo_v3](https://universe.roboflow.com/identvintern/groceries-9vwuo)
- [itens-de-dispensa-8pudf_v4](https://app.roboflow.com/ic-rfkuy/itens-de-dispensa-8pudf/4)

O arquivo `datasets/data.yaml` consolidado contém:

```yaml
names:
- Drinks
- Egg
- Juice
- Milk
- beverage
- food-box
- fruit
nc: 7
path: ./datasets
test:
- itens-de-dispensa-8pudf_v4/test/images
train:
- groceries-9vwuo_v3/train/images
- itens-de-dispensa-8pudf_v4/train/images
val:
- groceries-9vwuo_v3/valid/images
- itens-de-dispensa-8pudf_v4/valid/images
```

## Treinamento e Avaliação

O pipeline executa:
1. Download dos datasets
2. Geração do arquivo `data.yaml` consolidado
3. Treinamento do modelo YOLO (descongelando as últimas 5 camadas)
4. Avaliação quantitativa e visual no conjunto de teste
5. Salvamento de métricas, pesos e arquitetura

Exemplo de métricas obtidas (arquivo `resultados/metricas_teste.json`):

```json
{
	 "metrics/precision(B)": 0.24,
	 "metrics/recall(B)": 0.25,
	 "metrics/mAP50(B)": 0.19,
	 "metrics/mAP50-95(B)": 0.08,
	 "fitness": 0.08
}
```

Arquitetura do modelo salvo em `resultados/arquitetura.txt`.

## Resultados

Os resultados quantitativos e gráficos são salvos em `resultados/` e `resultados/predicoes/`, incluindo:
- Pesos do modelo treinado (`modelo_treinado.pt`)
- Métricas (`metricas.json`, `metricas_teste.json`)
- Gráficos de precisão, recall, F1, matriz de confusão
- Imagens de predição com bounding boxes

## Interface Gráfica

O arquivo `app.py` implementa uma interface KivyMD para visualizar e ajustar o inventário detectado. Permite:
- Carregar imagens e visualizar as detecções
- Ajustar manualmente quantidades detectadas
- Copiar resultados para a área de transferência

## Autores

Projeto desenvolvido por Christhian Costa Lima (202206840030) e Elen Cristina Rego Gomes (202206840014).

---
*Atualizado em: 15/09/2025*






