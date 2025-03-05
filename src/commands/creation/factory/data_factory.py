from src.enums.command_enum import CommandType
from src.commands.creation.factory.command_factory import CommandFactory

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
        
        elif command_type == CommandType.EXTRACT_FROM_GZ:
            self.check_required_arguments(kwargs, ["zip_file_path", "output_path"])
            return self.build_command(f"gunzip -c {kwargs['zip_file_path']} > {kwargs['output_path']}")
        
        elif command_type == CommandType.EXTRACT_FROM_ZIP:
            self.check_required_arguments(kwargs, ["zip_file_path", "first_file_to_extract", "second_file_to_extract", "dest_dir"])
            return self.build_command(f'unzip "{kwargs["zip_file_path"]}" "{kwargs["first_file_to_extract"]}" "{kwargs["second_file_to_extract"]}" -d "{kwargs["dest_dir"]}"')
        
        elif command_type == CommandType.REMOVE:
            self.check_required_arguments(kwargs, ["file_path"])
            return self.build_command(f'rm "{kwargs["file_path"]}"')

        # ********************* Commandes composées
        
        elif command_type == CommandType.DOWNLOAD_AND_EXTRACT_FROM_TAR:
            self.check_required_arguments(kwargs, ["url", "dest_dir"])

            tar_gz_file_path = kwargs["dest_dir"] + "/" +  kwargs["url"].split('/')[-1]
            folder_path = tar_gz_file_path[:-7] # Enlève l'extension ".tar.gz"

            if self.pathExists(folder_path):
                return self.build_command(f"echo \"📢 Le dossier {folder_path} existe déjà. Il n'est pas nécessaire de le retélécharger\"")

            commands = [
                self.create_command(CommandType.DOWNLOAD_FROM_URL, **kwargs),
                self.create_command(CommandType.EXTRACT_FROM_TAR, **dict(kwargs, file_path=tar_gz_file_path)),
            ]

            return self.build_composite_command(commands)
        
        elif command_type == CommandType.DOWNLOAD_AND_EXTRACT_FROM_GZ:
            self.check_required_arguments(kwargs, ["url", "dest_dir"])

            gz_file_path = kwargs["dest_dir"] + "/" +  kwargs["url"].split('/')[-1]
            output_path = gz_file_path[:-3] # Enlève l'extension ".gz"

            if self.pathExists(output_path):
                return self.build_command(f"echo \"📢 Le fichier {output_path} existe déjà. Il n'est pas nécessaire de le retélécharger\"")

            commands = [
                self.create_command(CommandType.DOWNLOAD_FROM_URL, **kwargs),
                self.create_command(CommandType.EXTRACT_FROM_GZ, **dict(kwargs, zip_file_path=gz_file_path, output_path=output_path)),
            ]

            return self.build_composite_command(commands)

        elif command_type == CommandType.DOWNLOAD_AND_EXTRACT_FROM_ZIP:
            self.check_required_arguments(kwargs, ["url", "dest_dir", "first_file_to_extract", "second_file_to_extract", "output_path"])

            zip_file_path = kwargs["dest_dir"] + "/" +  kwargs["url"].split('/')[-1]
            output_path = kwargs["output_path"]

            if self.pathExists(output_path):
                return self.build_already_exists_command(f"Le fichier: {output_path}")

            commands = [
                self.create_command(CommandType.DOWNLOAD_FROM_URL, **kwargs),
                self.create_command(CommandType.EXTRACT_FROM_ZIP, **dict(kwargs, zip_file_path=zip_file_path)),
                self.create_command(CommandType.REMOVE, **dict(file_path=zip_file_path))
            ]

            return self.build_composite_command(commands)
        
        else:
            raise ValueError(f"Commande inconnue pour DataGathering: {command_type}")


