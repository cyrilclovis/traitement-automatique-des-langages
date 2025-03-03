from src.commands.command import Command
from src.commands.composite_command import CompositeCommand
from src.commands.creation.factory.command_factory import CommandFactory
from src.enums.command_enum import CommandType

class Pipeline(Command):
    def __init__(self):
        # Utilisation de CompositeCommand pour gérer les commandes du pipeline
        self.commands = CompositeCommand()

    def add_command_from_factory(self, factory: CommandFactory, command_type: CommandType, **kwargs):
        """Ajoute une commande au pipeline."""
        self.commands.add_command(factory.create_command(command_type, **kwargs))
        return self  # Permet le chaînage des appels
    
    def add_command(self, command: Command):
        """Ajoute une commande au pipeline."""
        self.commands.add_command(command)
        return self  # Permet le chaînage des appels
    

    def execute(self):
        """Exécute les commandes du composite dans le pipeline."""
        print("🛠️ Début de l'exécution des commandes")
        self.commands.execute()
        print("✅ Toutes les commandes ont été exécutées.")
