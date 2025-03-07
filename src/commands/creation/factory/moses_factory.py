from src.enums.command_enum import CommandType
from src.commands.creation.factory.command_factory import CommandFactory
from src.utils.helpers import pathExists

class MosesCommandFactory(CommandFactory):
    """Factory pour la création des commandes associées au dépot Moses."""

    def check_moses_repository(self):
        """Vérifie si le dépôt mosesdecoder existe déjà."""
        if not pathExists("./data/mosesdecoder"):
            self.build_command("git clone https://github.com/moses-smt/mosesdecoder.git ./data/mosesdecoder").execute()

    def create_command(self, command_type: CommandType, **kwargs):

        self.check_moses_repository()

        # Vérifie l'existence du fichier de sortie avant de continuer
        if self.pathExists_kwargs("output_file", **kwargs):
            return self.build_already_exists_command(kwargs["output_file"])

        # ********************* Commandes simples
        
        if command_type == CommandType.TOKENIZE:
            self.check_required_arguments(kwargs, ["lang", "input_file", "output_file"])
            return self.build_command(
                f"./data/mosesdecoder/scripts/tokenizer/tokenizer.perl -l {kwargs['lang']} < {kwargs['input_file']} > {kwargs['output_file']}"
            )

        elif command_type == CommandType.TRAIN_TRUECASER_MODEL:
            self.check_required_arguments(kwargs, ["model_path", "corpus_path"])

            # Vérifie l'existence du modèle
            if self.pathExists(kwargs["model_path"]):
                return self.build_already_exists_command(kwargs["model_path"])
        
            return self.build_command(
                f"./data/mosesdecoder/scripts/recaser/train-truecaser.perl --model {kwargs['model_path']} --corpus {kwargs['corpus_path']}"
            )

        elif command_type == CommandType.TRUE_CASING:
            self.check_required_arguments(kwargs, ["model_path", "input_file", "output_file"])
            return self.build_command(
                f"./data/mosesdecoder/scripts/recaser/truecase.perl --model {kwargs['model_path']} < {kwargs['input_file']} > {kwargs['output_file']}"
            )
        
        elif command_type == CommandType.CLEAN_CORPUS:
            self.check_required_arguments(kwargs, ["input_file", "lang1", "lang2", "output_file", "min_len", "max_len"])

            # Vérifie l'existence du fichier de sortie avant de continuer
            if self.pathExists(f"{kwargs['input_file']}.{kwargs['lang1']}"):
                return self.build_already_exists_command(f"Les versions nettoyés de {kwargs['lang1']} et {kwargs['lang2']}: {kwargs['output_file']}")

            return self.build_command(
                f"./data/mosesdecoder/scripts/training/clean-corpus-n.perl {kwargs['input_file']} {kwargs['lang1']} {kwargs['lang2']} {kwargs['output_file']} {kwargs['min_len']} {kwargs['max_len']}"
            )
        
        else:
            raise ValueError(f"Commande inconnue pour TrueCasing: {command_type}")
