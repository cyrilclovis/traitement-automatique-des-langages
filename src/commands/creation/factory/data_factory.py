from src.enums.command_enum import CommandType
from src.commands.creation.factory.command_factory import CommandFactory

class DataGatheringCommandFactory(CommandFactory):
    """Factory pour la création des commandes concernant la collecte de données."""

    def create_command(self, command_type: CommandType, **kwargs):

        if command_type == CommandType.DOWNLOAD_FROM_URL:
            self.check_required_arguments(kwargs, ["url", "dest_dir"])
            return self.build_command(f"wget {kwargs['url']}", **{"-P": kwargs["dest_dir"]})
        
        elif command_type == CommandType.EXTRACT_FROM_TAR:
            self.check_required_arguments(kwargs, ["file_path", "dest_dir"])
            return self.build_command(f"tar", **{"xf": kwargs["file_path"], "-C": kwargs["dest_dir"]})
        
        elif command_type == CommandType.DOWNLOAD_AND_EXTRACT_FROM_TAR:
            self.check_required_arguments(kwargs, ["url", "dest_dir"])

            file_path = kwargs["dest_dir"] + "/" +  kwargs["url"].split('/')[-1]

            commands = [
                self.create_command(CommandType.DOWNLOAD_FROM_URL, **kwargs),
                self.create_command(CommandType.EXTRACT_FROM_TAR, **dict(kwargs, file_path=file_path)),
            ]

            return self.build_composite_command(commands)

        
        else:
            raise ValueError(f"Commande inconnue pour DataGathering: {command_type}")


