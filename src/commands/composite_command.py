from src.commands.command import Command
from typing import List

class CompositeCommand(Command):
    """Commande composite exécute une liste de commandes dans l’ordre."""
    
    def __init__(self, commands: List[Command] = None):
        self.commands: List[Command] = commands if commands is not None else []

    def add_command(self, command: Command):
        """Ajoute une commande à la liste."""
        self.commands.append(command)

    def execute(self):
        """Exécute chaque commande dans l'ordre."""
        for command in self.commands:
            command.execute()
