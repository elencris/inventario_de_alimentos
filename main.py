from download import baixar_datasets
from prepare_data import gerar_data_yaml
from train import treinar_modelo
from predict import avaliar_e_predizer
from config import NOME_MODELO, RESULTADOS_DIR

from ultralytics import YOLO
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    baixar_datasets()
    gerar_data_yaml()

    modelo = YOLO(NOME_MODELO)

    # ✅ Treina o modelo e retorna modelo treinado
    modelo_treinado = treinar_modelo(modelo)

    # ✅ Avalia o modelo treinado
    #avaliar_modelo(modelo_treinado)

    # ✅ Faz predições com o modelo treinado
    avaliar_e_predizer(modelo_treinado)

if __name__ == "__main__":
    main()
