from src.pipelines.pipeline import Pipeline
from src.enums.command_enum import CommandType

# Import des classes factory
from src.commands.creation.factory.data_factory import DataGatheringCommandFactory
from src.commands.creation.factory.open_nmt_factory import OpenNMTCommandFactory
from src.commands.creation.factory.corpus_factory import CorpusConstructionCommandFactory

class PipelineFactory:

    # Ce sont les factory depuis lesquels on récupère les commandes qui nous intéressent
    data_gathering_cmd_factory = DataGatheringCommandFactory()
    open_nmt_cmd_factory = OpenNMTCommandFactory()
    corpus_cmd_factory = CorpusConstructionCommandFactory()

    # ********************* Pipeline réutilisable
        
    @staticmethod
    def get_pipeline_for_corpuses_construction_from_tokenized_file(
        code: str = "en",
        nb_lines_for_train_corpus: int = 10000,
        nb_lines_for_dev_corpus: int = 1000,
        nb_lines_for_test_corpus: int = 500
    ) -> Pipeline:
        """
        Cette pipeline permet de :
        1. Télécharger et extraire un fichier compressé contenant un texte tokenisé.
        2. Construire trois corpus distincts à partir de ce fichier: TRAIN, DEV, TEST
        """
        pipeline = Pipeline()

        pipeline.add_command_from_factory(
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
            output_file_for_train_corpus=f"./data/europarl/Europarl_train_{format_k(nb_lines_for_train_corpus)}.tok.{code}",  # Fichier de sortie pour l'entraînement
            
            # ****** Commun au 2 futurs corpus (dev, test)
            starting_point=nb_lines_for_train_corpus,

            # Corpus dev
            nb_lines_to_extract_for_dev_corpus=nb_lines_for_dev_corpus,  # Nombre de lignes pour le corpus de validation
            output_file_for_dev_corpus=f"./data/europarl/Europarl_dev_{format_k(nb_lines_for_dev_corpus)}.tok.{code}",  # Fichier de sortie pour la validation
            
            # Corpus test
            nb_lines_to_extract_for_test_corpus=nb_lines_for_test_corpus,  # Nombre de lignes pour le corpus de test
            output_file_for_test_corpus=f"./data/europarl/Europarl_test_{format_k(nb_lines_for_test_corpus) }.tok.{code}",  # Fichier de sortie pour le test
            exclude_file=f"./data/europarl/Europarl_dev_{format_k(nb_lines_for_dev_corpus)}.tok.{code}"  # Fichier à exclure (corpus dev) pour le test
        )
        
        return pipeline


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

        pipeline.add_command(
            PipelineFactory.get_pipeline_for_corpuses_construction_from_tokenized_file("en")
        ).add_command(
            PipelineFactory.get_pipeline_for_corpuses_construction_from_tokenized_file("fr")
        )
        
        return pipeline


# ************************* UTLIS

# Fonction auxiliaire pour formater les chemins de fichiers
def format_k(x):
    return f"{x // 1000}k" if x >= 1000 else f"{x}"

# Format du nom du fichier avec le préfixe et le nombre de lignes
def generate_file_path(base_path: str, file_name: str = None, file_extension: str = None) -> str:
    build_file_name = f"{file_name}"

    if file_extension:
        build_file_name += f".{file_extension}"
    
    return f"{base_path}/{build_file_name}"