import yaml
from typing import Dict, List
from src.commands.command import Command

class ConfigCommand(Command):
    def __init__(self, config_file_path: str):
        """Initialisation avec des valeurs par défaut."""
        self.config_file_path = config_file_path
        self.config = {
            # ****************************** I) Construction du vocabulaire
            "src_vocab": "./data/toy-ende/run/vocab/src-vocab.txt",
            "tgt_vocab": "./data/toy-ende/run/vocab/tgt-vocab.txt",
            "overwrite": False,

            # Corpus concernés
            "data": {
                "corpus_1": {
                    "path_src": "./data/toy-ende/src-train.txt",
                    "path_tgt": "./data/toy-ende/tgt-train.txt"
                },
                "valid": {
                    "path_src": "./data/toy-ende/src-val.txt",
                    "path_tgt": "./data/toy-ende/tgt-val.txt"
                }
            },

            # ****************************** II) Entrainement du modèle
            # Utilise un GPU
            "world_size": 1,
            "gpu_ranks": [0],

            # Paramètres de sauvegarder et condition d'entrainement et évaluation (model, checkpoint et steps)
            "save_model": "./data/toy-ende/run/model",
            "save_data": "./data/toy-ende/run/model/checkpoints",
            "save_checkpoint_steps": 500,
            "train_steps": 1000,
            "valid_steps": 500,
        }

    def set_vocab_paths(self, src_vocab: str, tgt_vocab: str):
        self.config["src_vocab"] = src_vocab
        self.config["tgt_vocab"] = tgt_vocab
        return self  # Permet le chaînage

    def set_save_paths(self, save_data: str, save_model: str):
        self.config["save_data"] = save_data
        self.config["save_model"] = save_model
        return self

    def set_training_params(self, train_steps: int, valid_steps: int, checkpoint_steps: int):
        self.config["train_steps"] = train_steps
        self.config["valid_steps"] = valid_steps
        self.config["save_checkpoint_steps"] = checkpoint_steps
        return self

    def set_gpu_ranks(self, gpu_ranks: List[int]):
        self.config["gpu_ranks"] = gpu_ranks
        return self

    def add_corpus(self, corpus_name: str, path_src: str, path_tgt: str):
        """Ajoute un corpus personnalisé aux données d'entraînement."""
        self.config["data"][corpus_name] = {
            "path_src": path_src,
            "path_tgt": path_tgt
        }
        return self

    def build(self) -> Dict:
        """Retourne la configuration finale."""
        return self.config

    def execute(self):
        """Enregistre la configuration dans un fichier YAML."""
        with open(self.config_file_path, "w") as file:
            yaml.dump(self.config, file, default_flow_style=False)
        print(f"✅ Configuration enregistrée dans {self.config_file_path}")