from src.enums.command_enum import CommandType
from src.commands.creation.factory.command_factory import CommandFactory

class OpenNMTCommandFactory(CommandFactory):
    """Factory pour la création des commandes OpenNMT."""

    def create_command(self, command_type: CommandType, **kwargs):
        """Crée une commande spécifique pour OpenNMT."""
    
        if command_type == CommandType.YAML_CONFIG:
            self.check_required_arguments(kwargs, ["config_path"])
            return self.build_config_command(kwargs["config_path"])

        elif command_type == CommandType.BUILD_VOCAB:
            self.check_required_arguments(kwargs, ["config_path", "n_sample"])
            return self.build_command(self.build_vocab_command(**kwargs))
        
        elif command_type == CommandType.TRAIN:
            self.check_required_arguments(kwargs, ["config_path"])
            return self.build_command(self.train_command(**kwargs))
        
        elif command_type == CommandType.TRANSLATE:
            self.check_required_arguments(kwargs, ["model_path", "src_path", "output_path"])
            return self.build_command(self.translate_command(**kwargs))
        
        else:
            raise ValueError(f"Commande inconnue pour OpenNMT: {command_type}")


    def build_vocab_command(self, config_path: str, n_sample: int) -> str:
        """Crée la commande pour la génération du vocabulaire."""
        return f"onmt_build_vocab -config {config_path} -n_sample {n_sample}"


    def train_command(self, config_path: str) -> str:
        """Crée la commande pour entraîner le modèle."""
        return f"onmt_train -config {config_path}"


    def translate_command(self, model_path: str, src_path: str, output_path: str, gpu: int = 0, verbose: bool = False) -> str:
        """Crée la commande pour traduire avec le modèle."""
        command = f"onmt_translate -model {model_path} -src {src_path} -output {output_path} -gpu {gpu}"
        if verbose:
            command += " -verbose"
        return command
