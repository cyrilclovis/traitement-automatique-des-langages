from src.enums.command_enum import CommandType
from src.commands.creation.factory.command_factory import CommandFactory
from src.utils.helpers import pathExists

class DataGatheringCommandFactory(CommandFactory):
    """Factory pour la création des commandes concernant la collecte de données."""

    def create_command(self, command_type: CommandType, **kwargs):

        # ********************* Commandes simples

        if command_type == CommandType.DOWNLOAD_FROM_URL:
            self.check_required_arguments(kwargs, ["url", "dest_dir"])
            return self.build_command(f"wget {kwargs['url']}", **{"-P": kwargs["dest_dir"]})
        
        elif command_type == CommandType.EXTRACT_FROM_TAR:
            self.check_required_arguments(kwargs, ["file_path", "dest_dir"])
            return self.build_command(f"tar", **{"xf": kwargs["file_path"], "-C": kwargs["dest_dir"]})
        
        elif command_type == CommandType.EXTRACT_FROM_ZIP:
            self.check_required_arguments(kwargs, ["zip_file_path", "output_path"])
            return self.build_command(f"gunzip -c {kwargs['zip_file_path']} > {kwargs['output_path']}")

        # ********************* Commandes composées
        
        elif command_type == CommandType.DOWNLOAD_AND_EXTRACT_FROM_TAR:
            self.check_required_arguments(kwargs, ["url", "dest_dir"])

            tar_gz_file_path = kwargs["dest_dir"] + "/" +  kwargs["url"].split('/')[-1]
            folder_path = tar_gz_file_path[:-7] # Enlève l'extension ".tar.gz"

            if pathExists(folder_path):
                return self.build_command(f"echo \"📢 Le dossier {folder_path} existe déjà. Il n'est pas nécessaire de le retélécharger\"")

            commands = [
                self.create_command(CommandType.DOWNLOAD_FROM_URL, **kwargs),
                self.create_command(CommandType.EXTRACT_FROM_TAR, **dict(kwargs, file_path=tar_gz_file_path)),
            ]

            return self.build_composite_command(commands)
        
        elif command_type == CommandType.DOWNLOAD_AND_EXTRACT_FROM_ZIP:
            self.check_required_arguments(kwargs, ["url", "dest_dir"])

            zip_file_path = kwargs["dest_dir"] + "/" +  kwargs["url"].split('/')[-1]
            output_path = zip_file_path[:-3] # Enlève l'extension ".gz"

            if pathExists(output_path):
                return self.build_command(f"echo \"📢 Le fichier {output_path} existe déjà. Il n'est pas nécessaire de le retélécharger\"")

            commands = [
                self.create_command(CommandType.DOWNLOAD_FROM_URL, **kwargs),
                self.create_command(CommandType.EXTRACT_FROM_ZIP, **dict(kwargs, zip_file_path=zip_file_path, output_path=output_path)),
            ]

            return self.build_composite_command(commands)

        
        else:
            raise ValueError(f"Commande inconnue pour DataGathering: {command_type}")


