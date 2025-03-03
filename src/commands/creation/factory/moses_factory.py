from src.enums.command_enum import CommandType
from src.commands.creation.factory.command_factory import CommandFactory

class TrueCasingCommandFactory(CommandFactory):
    """Factory pour la création des commandes associées au dépot Moses."""

    def create_command(self, command_type: CommandType, **kwargs):

        # ********************* Commandes simples

        if command_type == CommandType.CLONE_MOSES:
            return self.build_command("git clone https://github.com/moses-smt/mosesdecoder.git")

        elif command_type == CommandType.TRAIN_TRUECASER_MODEL:
            self.check_required_arguments(kwargs, ["model_path", "corpus_path"])
            return self.build_command(
                f"/content/mosesdecoder/scripts/recaser/train-truecaser.perl --model {kwargs['model_path']} --corpus {kwargs['corpus_path']}"
            )

        elif command_type == CommandType.TRUE_CASING:
            self.check_required_arguments(kwargs, ["model_path", "input_file", "output_file"])
            return self.build_command(
                f"/content/mosesdecoder/scripts/recaser/truecase.perl --model {kwargs['model_path']} < {kwargs['input_file']} > {kwargs['output_file']}"
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
