from src.pipelines.pipeline import Pipeline
from src.enums.command_enum import CommandType

# Import des classes factory
from src.commands.creation.factory.data_factory import DataGatheringCommandFactory
from src.commands.creation.factory.open_nmt_factory import OpenNMTCommandFactory
from src.commands.creation.factory.corpus_factory import CorpusConstructionCommandFactory
from src.commands.creation.factory.moses_factory import MosesCommandFactory 

class PipelineFactory:

    # Ce sont les factory depuis lesquels on récupère les commandes qui nous intéressent
    data_gathering_cmd_factory = DataGatheringCommandFactory()
    open_nmt_cmd_factory = OpenNMTCommandFactory()
    corpus_cmd_factory = CorpusConstructionCommandFactory()
    moses_cmd_factory = MosesCommandFactory()

    # ********************* Pipelines réutilisables
    # Ce sont des pipelines qui regroupe des opérations que l'on sera amené à répéter; moyennant quoi, parfois, de nombreux paramètres doivent etre passés !
        
    @staticmethod
    def split_tokenized_corpus_into_train_dev_test(
        code: str,
        nb_lines_for_train_corpus: int = 10000,
        nb_lines_for_dev_corpus: int = 1000,
        nb_lines_for_test_corpus: int = 500
    ) -> tuple[Pipeline, dict]:
        """
        Cette pipeline permet de :
        1. Télécharger et extraire un fichier compressé contenant un texte tokenisé.
        2. Construire trois corpus distincts à partir de ce fichier: TRAIN, DEV, TEST

        Retourne :
        - La pipeline configurée.
        - Un dictionnaire contenant les chemins des fichiers de sortie.
        """

        train_file = f"./data/europarl/Europarl_train_{format_k(nb_lines_for_train_corpus)}.tok.{code}"
        dev_file = f"./data/europarl/Europarl_dev_{format_k(nb_lines_for_dev_corpus)}.tok.{code}"
        test_file = f"./data/europarl/Europarl_test_{format_k(nb_lines_for_test_corpus)}.tok.{code}"

        pipeline = Pipeline().add_command_from_factory(
            # Téléchargement des données tokenisés en anglais (=> on a le fichier en.tok)
            PipelineFactory.data_gathering_cmd_factory,
            CommandType.DOWNLOAD_AND_EXTRACT_FROM_ZIP,
            url=f"https://object.pouta.csc.fi/OPUS-Europarl/v8/mono/{code}.tok.gz", dest_dir="./data/europarl"
        ).add_command_from_factory(
            # ************************* On a obtenu deux fichiers tokenisés pour chacun d'entre eux, on crée les corpus TRAIN, DEV, TEST
            PipelineFactory.corpus_cmd_factory,
            CommandType.CORPUS_SPLITTING_INTO_TRAIN_DEV_TEST_CORPUSES,
            # ****** Commun à tous
            tokenized_corpus_path=f"./data/europarl/{code}.tok",

            # Corpus train
            nb_lines_to_extract_for_train_corpus=nb_lines_for_train_corpus,  # Nombre de lignes à extraire pour le corpus d'entraînement
            output_file_for_train_corpus=train_file,  # Fichier de sortie pour l'entraînement
            
            # ****** Commun au 2 futurs corpus (dev, test)
            starting_point=nb_lines_for_train_corpus,

            # Corpus dev
            nb_lines_to_extract_for_dev_corpus=nb_lines_for_dev_corpus,  # Nombre de lignes pour le corpus de validation
            output_file_for_dev_corpus=dev_file,  # Fichier de sortie pour la validation
            
            # Corpus test
            nb_lines_to_extract_for_test_corpus=nb_lines_for_test_corpus,  # Nombre de lignes pour le corpus de test
            output_file_for_test_corpus=test_file,  # Fichier de sortie pour le test
            exclude_file=dev_file  # Fichier à exclure (corpus dev) pour le test
        )
        
        return pipeline, {
            "train": train_file,
            "dev": dev_file,
            "test": test_file
        }
    
    def train_and_apply_truecasing(code: str, model_path: str, files_names: dict) -> Pipeline:
        """
        Entraîne le modèle Truecaser et l'applique sur les corpus TRAIN, DEV et TEST.

        Args:
            code: Code de la langue (ex: en, fr)
            model_path (str): Le chemin où sauvegarder le modèle de Truecasing.
            files_names (dict): Dictionnaire contenant les chemins des fichiers (train, dev, test).

        Returns:
            Pipeline: La pipeline mise à jour avec les étapes d'entraînement et d'application du Truecasing.
        """

        # *********** On entraine le modèle
        pipeline = Pipeline().add_command_from_factory(
            PipelineFactory.moses_cmd_factory,
            CommandType.SOLVE_DEPENDENCIES_AND_TRAIN_TRUECASER_MODEL,
            model_path=model_path,
            corpus_path=files_names["train"]
        )

        # *********** Application du Truecasing sur TRAIN, DEV et TEST
        truecased_files_names = {}

        for split in ["train", "dev", "test"]:
            output_file = insert_before_extension(files_names[split], ".true", str(code))

            pipeline.add_command_from_factory(
                PipelineFactory.moses_cmd_factory,
                CommandType.TRUE_CASING,
                model_path=model_path,
                input_file=files_names[split],
                output_file=output_file
            )

            # Enregistre le nom du fichier transformé dans le dictionnaire
            truecased_files_names[split] = output_file

        return pipeline, truecased_files_names

    # ********************* Pipelines spécifiques

    @staticmethod
    def get_pipeline_i1(n_sample: int = 1000) -> Pipeline:
        """
        [PIPELINE I.1 => Expérimentation]
        """
        pipeline = Pipeline()

        pipeline.add_command_from_factory(
            # On télécharge les données depuis internet
            PipelineFactory.data_gathering_cmd_factory,
            CommandType.DOWNLOAD_AND_EXTRACT_FROM_TAR,
            url="https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz", dest_dir="./data"
        ).add_command_from_factory(
            # On crée le fichier YAML
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.YAML_CONFIG,
            config_path="./config/toy-ende.yaml"
        ).add_command_from_factory(
            # On construit le vocabulaire
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.BUILD_VOCAB,
            config_path="./config/toy-ende.yaml", n_sample=n_sample
        ).add_command_from_factory(
            # On fait l'entraiement
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRAIN,
            config_path="./config/toy-ende.yaml"
        ).add_command_from_factory(
            # On traduit
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRANSLATE,
            model_path = f"./data/toy-ende/run/model_step_{n_sample}.pt",
            src_path = "./data/toy-ende/src-test.txt",
            output_path = f"./data/toy-ende/run/pred_{n_sample}.txt"
        )
        return pipeline
    
    @staticmethod
    def get_pipeline_i2() -> Pipeline:
        """
        [PIPELINE I.2]
        """
        pipeline = Pipeline()
        language_codes = ["fr", "en"]

        for lang_code in language_codes:

            # ***** Pour chaque code, on met en place les corpus, on entraine le modèle puis on l'applique sur nos corpus.
            build_corpora_pipeline, files_names = PipelineFactory.split_tokenized_corpus_into_train_dev_test(lang_code)
            model_path = f"./data/europarl/truecase-model.{lang_code}"
            train_and_apply_casing_pipeline, truecased_files_names = PipelineFactory.train_and_apply_truecasing(lang_code, model_path, files_names)

            pipeline.add_command(build_corpora_pipeline).add_command(train_and_apply_casing_pipeline)

        # ***** Application du nettoyage sur TRAIN, DEV et TEST pour le couple situé dans language_codes.
        for split in ["train", "dev", "test"]:

            input_file = remove_part_from_filename(truecased_files_names[split], f".{lang_code}") # On enlève le ".{lang_code}", ici .en ou .fr

            pipeline.add_command_from_factory(
                PipelineFactory.moses_cmd_factory,
                CommandType.CLEAN_CORPUS,
                input_file=input_file,
                lang1=language_codes[0],
                lang2=language_codes[1],
                output_file=f"{input_file}.clean",
                min_len=1,
                max_len=80
            )
     
        return pipeline


# ************************* UTLIS

# Fonction auxiliaire pour formater les chemins de fichiers
def format_k(x):
    return f"{x // 1000}k" if x >= 1000 else f"{x}"

# Permet d'insérer un suffix avant l'extension. Cela est très utile car le respect des noms de fichier
# est important pour l'utilisation des commandes de moses.
def insert_before_extension(file_name: str, suffix: str, extension: str) -> str:
    if extension not in file_name:
        raise ValueError(f"'{extension}' n'est pas dans '{file_name}'")
    return file_name.replace(f".{extension}", f"{suffix}.{extension}", 1)

def remove_part_from_filename(file_name: str, part_to_remove: str) -> str:
    if part_to_remove not in file_name:
        raise ValueError(f"'{part_to_remove}' n'est pas dans' '{file_name}'")
    return file_name.replace(f"{part_to_remove}", "", 1)

