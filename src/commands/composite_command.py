from src.commands.command import Command
from typing import List

class CompositeCommand(Command):
    """Commande composite exécute une liste de commandes dans l’ordre."""
    
    def __init__(self):
        self.commands: List[Command] = []

    def add_command(self, command: Command):
        """Ajoute une commande à la liste."""
        self.commands.append(command)

    def execute(self):
        """Exécute chaque commande dans l'ordre."""
        print("🛠️ Début de l'exécution des commandes")
        for command in self.commands:
            command.execute()
        print("✅ Toutes les commandes ont été exécutées.")