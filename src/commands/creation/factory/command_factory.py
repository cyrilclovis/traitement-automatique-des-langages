from abc import ABC, abstractmethod
from typing import List

from src.enums.command_enum import CommandType
from src.commands.command import Command
from src.commands.config_command import ConfigCommand
from src.commands.composite_command import CompositeCommand
from src.commands.creation.command_builder import CommandBuilder
from src.utils.helpers import pathExists

class CommandFactory(ABC):
    """Classe abstraite pour créer des commandes spécifiques."""

    @abstractmethod
    def create_command(self, command_type: CommandType, **kwargs):
        """Doit être implémentée par les sous-classes."""
        raise NotImplementedError("Cette méthode doit être implémentée par une sous-classe.")
    
    def check_required_arguments(self, kwargs, required_args):
        """Vérifie la présence des arguments nécessaires dans kwargs."""
        missing_args = [arg for arg in required_args if arg not in kwargs or kwargs[arg] is None]
        
        if missing_args:
            raise ValueError(f"⚠️ Les arguments suivantes sont absents ou null: {', '.join(missing_args)}")
        
    def pathExists_kwargs(self, key: str, **kwargs):
        """Vérifie l'existence d'un fichier ou d'un dossier"""
        if key in kwargs:
            path_to_check = kwargs[key]
            return pathExists(path_to_check)
        return False
    
    def pathExists(self, key: str):
        return pathExists(key)

    # ****** Les différents type de commande que factory peut renvoyer
    def build_already_exists_command(self, already_exists_path: str):
        """Construit la commande permettant d'afficher sur le terminal l'existence d'un fichier"""
        return self.build_command(f"echo \"📢 {already_exists_path} existe déjà.\"")

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
