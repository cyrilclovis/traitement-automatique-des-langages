from enum import Enum

class CommandType(Enum):
    # ********************* Commandes pour la collecte de données
    DOWNLOAD_FROM_URL = "download_from_url"
    EXTRACT_FROM_TAR = "extract_from_tar"
    EXTRACT_FROM_ZIP = "extract_from_zip"

    DOWNLOAD_AND_EXTRACT_FROM_TAR = "download_and_extract_from_tar"
    DOWNLOAD_AND_EXTRACT_FROM_ZIP = "download_and_extract_from_zip"

    # ********************* Commandes pour les créations de corpus

    EXTRACT_FIRST_N_LINES = "extract_first_n_lines"
    EXTRACT_FIRST_N_LINES_FROM_STARTING_POINT = "extract_first_n_lines_from_starting_point"
    EXTRACT_N_RANDOM_LINES_FROM_STARTING_POINT = "extract_n_random_lines_from_starting_point"
    EXTRACT_N_RANDOM_LINES_WHICH_ARE_NOT_IN_GIVEN_FILE = "extract_n_random_lines_which_are_not_in_given_file"

    CORPUS_SPLITTING_INTO_TRAIN_DEV_TEST_CORPUSES = "corpus_splitting_into_train_dev_test_corpuses"

    # ********************* Commandes pour la partie Moses

    CLONE_MOSES = "clone_moses"
    TRAIN_TRUECASER_MODEL = "train_truecaser_model"
    TRUE_CASING = "true_casing"
    CLEAN_CORPUS = "clean_corpus"

    SOLVE_DEPENDENCIES_AND_TRAIN_TRUECASER_MODEL = "solve_dependencies_and_train_truecaser_model"

    # ********************* Commandes pour les configurations
    YAML_CONFIG = "yaml_config"
    
    # ********************* Commandes pour le prétraitement
    TOKENIZE = "tokenize"
    BUILD_VOCAB = "build_vocab"

    # ********************* Commandes pour l'entraînement
    TRAIN = "train"
    TRANSLATE = "translate"

    # ********************* Commandes pour l'évaluation
    

