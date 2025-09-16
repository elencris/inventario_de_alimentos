from download import baixar_datasets
from prepare_data import gerar_data_yaml
from train import treinar_modelo
from predict import avaliar_e_predizer
from config import NOME_MODELO

from ultralytics import YOLO


def main() -> None:
    """Executa o fluxo principal do programa.

    Este pipeline realiza as seguintes etapas:
    1. Baixa os datasets do Roboflow.
    2. Gera o arquivo `data.yaml` unificado para o YOLO.
    3. Inicializa e treina o modelo YOLO com os dados.
    4. Avalia o modelo treinado e realiza predições no conjunto de teste.

    Parâmetros
    ----------
    Nenhum

    Retorno
    -------
    None
        Esta função não retorna nenhum valor. Executa o fluxo completo do projeto.
    """
    
    baixar_datasets()
    gerar_data_yaml()

    modelo = YOLO(NOME_MODELO)
    modelo_treinado = treinar_modelo(modelo)

    avaliar_e_predizer(modelo_treinado)


if __name__ == "__main__":
    main()