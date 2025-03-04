from abc import ABC, abstractmethod
from typing import List

from src.enums.command_enum import CommandType
from src.commands.command import Command
from src.commands.config_command import ConfigCommand
from src.commands.composite_command import CompositeCommand
from src.commands.creation.command_builder import CommandBuilder

class CommandFactory(ABC):
    """Classe abstraite pour créer des commandes spécifiques."""

    @abstractmethod
    def create_command(self, command_type: CommandType, **kwargs):
        """Doit être implémentée par les sous-classes."""
        raise NotImplementedError("Cette méthode doit être implémentée par une sous-classe.")
    
    def check_required_arguments(self, kwargs, required_args):
        """Checks if all required arguments are present in kwargs."""
        missing_args = [arg for arg in required_args if arg not in kwargs or kwargs[arg] is None]
        
        if missing_args:
            raise ValueError(f"⚠️ Les arguments suivantes sont absents ou null: {', '.join(missing_args)}")

    # ****** Les différents type de commande que factory peut renvoyer

    def build_command(self, base_command: str, **kwargs):
        """Crée une commande en appelant la méthode de CommandBuilder."""
        return CommandBuilder.build_command(base_command, **kwargs)
    
    def build_composite_command(self, commands: List[Command]) -> CompositeCommand:
        """Crée un CompositeCommand à partir d'une liste de commandes."""
        return CompositeCommand(commands)
    
    def build_config_command(self, config_file_path: str) -> ConfigCommand:
        """Crée une commande de configuration."""
        # TODO: revenir dessus, car là on délègue à l'utilisateur la manière dont les paramètres de ConfigCommand évolue.
        return ConfigCommand(config_file_path)
