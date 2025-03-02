from enum import Enum

class CommandType(Enum):
    # ********************* Commandes pour la collecte de données
    DOWNLOAD_FROM_URL = "download_from_url"
    EXTRACT_FROM_TAR = "extract_from_tar"
    DOWNLOAD_AND_EXTRACT_FROM_TAR = "download_and_extract_from_tar"
    
    # ********************* Commandes pour le prétraitement
    TOKENIZE = "tokenize"
    BUILD_VOCAB = "build_vocab"

    # ********************* Commandes pour l'entraînement
    TRAIN = "train"

    # ********************* Commandes pour l'évaluation
    

