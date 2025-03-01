import subprocess
from abc import ABC, abstractmethod

import threading

class CommandExecutor:
    """Exécute une commande shell et affiche la sortie en temps réel."""
   
    @staticmethod
    def run(command: str):
        print(f"🔹 Exécution : {command}")


        def stream_output(pipe, is_error=False):
            """Lit et affiche la sortie du processus en temps réel."""
            for line in iter(pipe.readline, ''):
                output = line.strip()
                if output:
                    print(output)
            pipe.close()


        try:
            process = subprocess.Popen(
                command,
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



class OpenNMTTask(ABC):
    """Interface pour définir une tâche OpenNMT."""

    def execute(self):
        """Exécute la tâche OpenNMT."""
        CommandExecutor.run(self.command)

class BuildVocabTask(OpenNMTTask):
    """Tâche pour générer le vocabulaire."""

    def __init__(self, config_path: str, n_sample: int = 10000):
        self.command = f"onmt_build_vocab -config {config_path} -n_sample {n_sample}"


class TrainTask(OpenNMTTask):
    """Tâche pour entraîner le modèle."""

    def __init__(self, config_path: str):
        self.command = f"onmt_train -config {config_path}"
        print(self.command)


class TranslateTask(OpenNMTTask):
    """Tâche pour traduire un fichier source."""

    def __init__(self, model_path: str, src_path: str, output_path: str, gpu: int = 0, verbose: bool = False):
        self.command = f"onmt_translate -model {model_path} -src {src_path} -output {output_path} -gpu {gpu}"

        print(self.command)
        if verbose:
            self.command += " -verbose"

class OpenNMTPipeline:
    """Gère l'exécution des différentes tâches OpenNMT en chaîne."""

    def __init__(self):
        self.tasks = []

    def add_task(self, task: OpenNMTTask):
        """Ajoute une tâche à la liste des exécutions."""
        self.tasks.append(task)

    def run(self):
        """Exécute toutes les tâches dans l'ordre."""
        for task in self.tasks:
            task.execute()

# 🛠️ Exemple d'utilisation
if __name__ == "__main__":
    config_path = "./config/toy-ende.yaml"
    model_path = "./data/toy-ende/run/model_step_1000.pt"
    src_path = "./data/toy-ende/src-test.txt"
    output_path = "./data/toy-ende/pred_1000.txt"

    pipeline = OpenNMTPipeline()
    pipeline.add_task(BuildVocabTask(config_path))
    pipeline.add_task(TrainTask(config_path))
    pipeline.add_task(TranslateTask(model_path, src_path, output_path, gpu=0, verbose=True))

    pipeline.run()
