from src.pipelines.pipeline import Pipeline
# Import des classes factory
from src.commands.creation.factory.data_factory import DataGatheringCommandFactory
from src.commands.creation.factory.open_nmt_factory import OpenNMTCommandFactory

from src.enums.command_enum import CommandType

class PipelineFactory:

    # Ce sont les factory depuis lesquels on récupère les commandes qui nous intéressent
    data_gathering_cmd_factory = DataGatheringCommandFactory()
    open_nmt_cmd_factory = OpenNMTCommandFactory()

    @staticmethod
    def get_classic_pipeline() -> Pipeline:
        """Renvoie un pipeline classique (initial), avec les commandes prédéfinies."""
        pipeline = Pipeline()

        pipeline.add_command(
            # On télécharge les données depuis internet
            PipelineFactory.data_gathering_cmd_factory,
            CommandType.DOWNLOAD_AND_EXTRACT_FROM_TAR,
            url="https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz", dest_dir="./data"
        ).add_command(
            # On crée le fichier YAML
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.YAML_CONFIG,
            config_path="./config/toy-ende.yaml"
        ).add_command(
            # On construit le vocabulaire
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.BUILD_VOCAB,
            config_path="./config/toy-ende.yaml", n_sample=str(10000)
        ).add_command(
            # On fait l'entraiement
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRAIN,
            config_path="./config/toy-ende.yaml"
        ).add_command(
            # On traduit
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRANSLATE,
            model_path = "./data/toy-ende/run/model_step_1000.pt",
            src_path = "./data/toy-ende/src-test.txt",
            output_path = "./data/toy-ende/pred_1000.txt"
        )
        return pipeline
