from src.enums.command_enum import CommandType
from src.commands.creation.factory.command_factory import CommandFactory
from src.utils.helpers import pathExists

class MosesCommandFactory(CommandFactory):
    """Factory pour la création des commandes associées au dépot Moses."""

    def create_command(self, command_type: CommandType, **kwargs):

        # ********************* Commandes simples

        if command_type == CommandType.CLONE_MOSES:

            if pathExists("./data/mosesdecoder"):
                return self.build_command(f"echo \"📢 Le dépôt mosesdecoder existe déjà. Il n'est pas nécessaire de le recloner\"")
            return self.build_command("git clone https://github.com/moses-smt/mosesdecoder.git ./data/mosesdecoder")

        elif command_type == CommandType.TRAIN_TRUECASER_MODEL:
            self.check_required_arguments(kwargs, ["model_path", "corpus_path"])
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
            return self.build_command(
                f"./data/mosesdecoder/scripts/training/clean-corpus-n.perl {kwargs['input_file']} {kwargs['lang1']} {kwargs['lang2']} {kwargs['output_file']} {kwargs['min_len']} {kwargs['max_len']}"
            )
        
        # ********************* Commandes composées

        elif command_type == CommandType.SOLVE_DEPENDENCIES_AND_TRAIN_TRUECASER_MODEL:
            self.check_required_arguments(kwargs, ["model_path", "corpus_path"])

            commands = [
                self.create_command(CommandType.CLONE_MOSES),
                self.create_command(CommandType.TRAIN_TRUECASER_MODEL, **kwargs),
            ]

            return self.build_composite_command(commands)

        else:
            raise ValueError(f"Commande inconnue pour TrueCasing: {command_type}")
