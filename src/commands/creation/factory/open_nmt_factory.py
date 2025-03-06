from src.enums.command_enum import CommandType
from src.commands.creation.factory.command_factory import CommandFactory

class OpenNMTCommandFactory(CommandFactory):
    """Factory pour la création des commandes OpenNMT."""

    def create_command(self, command_type: CommandType, **kwargs):
        """Crée une commande spécifique pour OpenNMT."""
    
        if command_type == CommandType.YAML_CONFIG:
            self.check_required_arguments(kwargs, ["config_path"])

            # Vérifie l'existence du fichier du YAML
            if self.pathExists(kwargs["config_path"]):
                return self.build_already_exists_command(kwargs["config_path"])
            return self.build_config_command(kwargs["config_path"])

        elif command_type == CommandType.BUILD_VOCAB:
            self.check_required_arguments(kwargs, ["config_path", "src_vocab", "tgt_vocab"])

            if self.pathExists(kwargs["src_vocab"]) and self.pathExists(kwargs["tgt_vocab"]):
                return self.build_composite_command([
                    self.build_already_exists_command("Le vocabulaire source: " + kwargs["src_vocab"]),
                    self.build_already_exists_command("Le vocabulaire cible: " + kwargs["tgt_vocab"]),
                ])

            command = f"onmt_build_vocab -config {kwargs['config_path']}"
    
            if "n_sample" in kwargs and kwargs["n_sample"] is not None:
                command += f" -n_sample {kwargs['n_sample']}"

            return self.build_command(command)

        
        elif command_type == CommandType.TRAIN:
            self.check_required_arguments(kwargs, ["config_path", "model_path"])

            # Vérifie l'existence du modèle
            if self.pathExists(kwargs["model_path"]):
                return self.build_already_exists_command("Le modèle: " + kwargs["model_path"])
            return self.build_command(f"onmt_train -config {kwargs['config_path']}")
        
        elif command_type == CommandType.TRANSLATE:
            self.check_required_arguments(kwargs, ["model_path", "src_path", "output_path"])

            # Vérifie l'existence du fichier YAML
            if self.pathExists(kwargs["output_path"]):
                return self.build_already_exists_command("La traduction: " + kwargs["output_path"])
            return self.build_command(self.translate_command(**kwargs))
        
        elif command_type == CommandType.BLEU_SCORE:
            self.check_required_arguments(kwargs, ["reference_file", "translation_file"])
            return self.build_command(
                f"perl ./data/multi_bleu.pl {kwargs['reference_file']} < {kwargs['translation_file']}"
            )
        
        else:
            raise ValueError(f"Commande inconnue pour OpenNMT: {command_type}")


    def translate_command(self, model_path: str, src_path: str, output_path: str, gpu: int = 0, verbose: bool = False) -> str:
        """Crée la commande pour traduire avec le modèle."""
        command = f"onmt_translate -model {model_path} -src {src_path} -output {output_path} -gpu {gpu}"
        if verbose:
            command += " -verbose"
        return command
