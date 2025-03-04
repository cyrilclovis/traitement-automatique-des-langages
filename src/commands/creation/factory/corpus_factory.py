from src.enums.command_enum import CommandType
from src.commands.creation.factory.command_factory import CommandFactory
from src.utils.helpers import pathExists

class CorpusConstructionCommandFactory(CommandFactory):
    """Factory pour la création des commandes concernant la construction de corpus."""

    def create_command(self, command_type: CommandType, **kwargs):

        # ********************* Commandes simples
        
        if command_type == CommandType.EXTRACT_FIRST_N_LINES:
            self.check_required_arguments(kwargs, ["nb_lines_to_extract", "file_path", "output_file"])
            return self.build_command(f"head -n {kwargs['nb_lines_to_extract']} {kwargs['file_path']} > {kwargs['output_file']}")

        elif command_type == CommandType.EXTRACT_N_RANDOM_LINES_FROM_STARTING_POINT:
            self.check_required_arguments(kwargs, ["starting_point", "file_path", "nb_lines_to_extract", "output_file"])
            return self.build_command(f"tail -n +{int(kwargs['starting_point'])+1} {kwargs['file_path']} | shuf -n {kwargs['nb_lines_to_extract']} > {kwargs['output_file']}")

        elif command_type == CommandType.EXTRACT_N_RANDOM_LINES_WHICH_ARE_NOT_IN_GIVEN_FILE:
            self.check_required_arguments(kwargs, ["starting_point", "file_path", "output_file", "exclude_file", "nb_lines_to_extract"])
            return self.build_command(f"tail -n +{int(kwargs['starting_point'])+1} {kwargs['file_path']} | grep -vxF -f {kwargs['exclude_file']} | shuf -n {kwargs['nb_lines_to_extract']} > {kwargs['output_file']}")
        
        # ********************* Commandes composées

        elif command_type == CommandType.CORPUS_SPLITTING_INTO_TRAIN_DEV_TEST_CORPUSES:
            self.check_required_arguments(kwargs,
                [
                 "tokenized_corpus_path", # Commun à tous
                 
                 # Pour la mise en place du corpus d'entrainement
                 "nb_lines_to_extract_for_train_corpus",
                 "output_file_for_train_corpus",

                 # Commun aux deux autres futurs corpus (dev, test)
                 "starting_point",

                 # Pour la mise en place du corpus dev
                 "nb_lines_to_extract_for_dev_corpus",
                 "output_file_for_dev_corpus",

                 # Pour la mise en place du corpus test
                 "nb_lines_to_extract_for_test_corpus",
                 "output_file_for_test_corpus",
                 "exclude_file",

                ])

            commands = [
                # Train + Vérifier si le fichier existe déjà
                self.build_command(f"echo \"📢 Le fichier {kwargs['output_file_for_train_corpus']} existe déjà. Il n'est pas nécessaire de le recréer\"")
                if pathExists(kwargs["output_file_for_train_corpus"])
                else self.create_command(
                    CommandType.EXTRACT_FIRST_N_LINES,
                    nb_lines_to_extract=kwargs["nb_lines_to_extract_for_train_corpus"],
                    file_path=kwargs["tokenized_corpus_path"],
                    output_file=kwargs["output_file_for_train_corpus"]
                ),
                # Dev + Vérifier si le fichier existe déjà
                self.build_command(f"echo \"📢 Le fichier {kwargs['output_file_for_dev_corpus']} existe déjà. Il n'est pas nécessaire de le recréer\"")
                if pathExists(kwargs["output_file_for_dev_corpus"])
                else self.create_command(
                    CommandType.EXTRACT_N_RANDOM_LINES_FROM_STARTING_POINT,
                    starting_point=kwargs["starting_point"],
                    file_path=kwargs["tokenized_corpus_path"],
                    nb_lines_to_extract=kwargs["nb_lines_to_extract_for_dev_corpus"],
                    output_file=kwargs["output_file_for_dev_corpus"]
                ),
                # Test + Vérifier si le fichier existe déjà
                self.build_command(f"echo \"📢 Le fichier {kwargs['output_file_for_test_corpus']} existe déjà. Il n'est pas nécessaire de le recréer\"")
                if pathExists(kwargs["output_file_for_test_corpus"])
                else self.create_command(
                    CommandType.EXTRACT_N_RANDOM_LINES_WHICH_ARE_NOT_IN_GIVEN_FILE,
                    starting_point=kwargs["starting_point"],
                    file_path=kwargs["tokenized_corpus_path"],
                    nb_lines_to_extract=kwargs["nb_lines_to_extract_for_test_corpus"],
                    output_file=kwargs["output_file_for_test_corpus"],
                    exclude_file=kwargs["exclude_file"]
                )
            ]

            return self.build_composite_command(commands)

        else:
            raise ValueError(f"Commande inconnue pour CorpusConstruction: {command_type}")
