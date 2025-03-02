from typing import Optional
from src.commands.shell_command import ShellCommand

class CommandBuilder:

    @staticmethod
    def build_command(base_command: str, **kwargs):
        """Construit une commande avec les options données."""
        builder = CommandBuilder(base_command)

        # Ajouter des options spécifiques à la commande
        for option_name, option_value in kwargs.items():
            builder.set_option(option_name, option_value)

        # Retourne la commande finale construite
        return builder.build()


    def __init__(self, base_command: str):
        self.base_command = base_command
        self.options = {}


    def set_option(self, key: str, value: Optional[str]):
        """Ajoute une option à la commande."""
        if value is not None:
            self.options[key] = value
        return self  # Permet le chaînage


    def build(self) -> ShellCommand:
        """Construit la commande finale."""
        cmd_str = self.base_command
        for key, value in self.options.items():
            cmd_str += f" {key} {value}"
        return ShellCommand(cmd_str)
