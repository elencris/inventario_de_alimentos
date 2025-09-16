"""
Configuração centralizada do logging para o projeto.

Este módulo configura o logger padrão para registrar mensagens tanto em
arquivo quanto no console. O arquivo de log é salvo dentro do diretório
de resultados definido no arquivo de configuração `config.py`.

Configurações:
- Nível de log: INFO
- Formato: timestamp, nível do log e mensagem
- Handlers:
    - FileHandler: escreve logs no arquivo 'app.log' em modo append
    - StreamHandler: imprime logs no console

Variáveis exportadas:
- logger: objeto logger raiz configurado para uso em outros módulos

Uso:
Importe o logger deste módulo para registrar mensagens uniformemente:
    from logger import logger
    logger.info("Mensagem de informação")
"""

import logging
import os
from config import RESULTADOS_DIR

# Garante que o diretório de resultados exista
os.makedirs(RESULTADOS_DIR, exist_ok=True)

LOG_FILE = os.path.join(RESULTADOS_DIR, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()
