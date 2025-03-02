import subprocess
import threading

from src.commands.command import Command

class ShellCommand(Command):
    """Exécute une commande shell et affiche la sortie en temps réel."""

    def __init__(self, command: str):
        self.command = command

    def execute(self):
        print(f"🔹 Exécution : {self.command}")

        def stream_output(pipe, is_error=False):
            """Lit et affiche la sortie du processus en temps réel."""
            for line in iter(pipe.readline, ''):
                output = line.strip()
                if output:
                    print(output)
            pipe.close()

        try:
            process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # Threads pour capturer stdout et stderr en temps réel
            stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,))
            stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, True))

            stdout_thread.start()
            stderr_thread.start()

            # Attendre la fin du processus
            process.wait()

            stdout_thread.join()
            stderr_thread.join()

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"❌ Erreur lors de l'exécution de la commande:\n{e}")