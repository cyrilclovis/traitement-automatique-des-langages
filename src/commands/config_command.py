import yaml
from typing import Dict, List
from src.commands.command import Command

class ConfigCommand(Command):
    
    required_fields = [
        # Pour le vocabulaire
        "src_vocab", "tgt_vocab",

        # Pour l'entrainement
        "save_model", "save_data"
    ]


    def __init__(self, folder_base_path: str, clean_truecased_files_names: dict, language_codes: List[str], config_file_path: str):
        """Initialisation avec des valeurs par défaut."""
        self.config_file_path = config_file_path
        self.config = {
            # ****************************** I) Construction du vocabulaire
            "src_vocab": "",
            "tgt_vocab": "",
            "overwrite": False,

            # Corpus concernés
            "data": {
                "corpus_1": {
                    "path_src": "",
                    "path_tgt": ""
                },
                "valid": {
                    "path_src": "",
                    "path_tgt": ""
                }
            },

            # ****************************** II) Entrainement du modèle
            # Utilise un GPU
            "world_size": 1,

            # Paramètres de sauvegarder et condition d'entrainement et évaluation (model, checkpoint et steps)
            "save_model": "",
            "save_data": "",
            "save_checkpoint_steps": 500,
            "train_steps": 1000,
            "valid_steps": 500,
        }

        self.add_cleaned_corpus_path(clean_truecased_files_names, language_codes).set_all_required_path(folder_base_path)


    def set_training_params(self, train_steps: int, valid_steps: int, checkpoint_steps: int):
        self.config["train_steps"] = train_steps
        self.config["valid_steps"] = valid_steps
        self.config["save_checkpoint_steps"] = checkpoint_steps
        return self

    def set_gpu_ranks(self, gpu_ranks: List[int]):
        self.config["gpu_ranks"] = gpu_ranks
        return self

    def add_cleaned_corpus_path(self, clean_truecased_files_names: dict, language_codes: List[str]):
        """Ajout les chemins vers les corpus."""

        # Mapping des clés pour correspondre aux bons noms de configuration
        key_mapping = {
            "train": "corpus_1",
            "dev": "valid"
        }

        # Fonction anonyme pour obtenir la clé correspondante
        get_mapped_key = lambda key: key_mapping.get(key)

        for key, file_name in clean_truecased_files_names.items():
            mapped_key = get_mapped_key(key)
            if mapped_key:  # Ignore "test"
                self.config["data"][mapped_key] = {
                    "path_src": f"{file_name}.{language_codes[0]}",
                    "path_tgt": f"{file_name}.{language_codes[1]}"
                }

        return self

    
    def set_all_required_path(self, base_path: str):
        for required_field in ConfigCommand.required_fields:
            self.config[required_field] += f"{base_path}/run" + self.complete_path(required_field)
        return self
    
    def complete_path(self, required_field):
        key_mapping = {
            "src_vocab": "/vocab.src",
            "tgt_vocab": "/vocab.tgt",
            "save_model": "/model",
            "save_data": "/samples",
        }

        return key_mapping.get(required_field)
    
    def get_vocab_source_path(self):
        return self.config["src_vocab"]
    
    def get_vocab_target_path(self):
        return self.config["tgt_vocab"]


    def build(self) -> Dict:
        """Retourne la configuration finale."""
        return self.config

    def execute(self):
        """Enregistre la configuration dans un fichier YAML."""
        with open(self.config_file_path, "w") as file:
            yaml.dump(self.config, file, default_flow_style=False)
        print(f"✅ Configuration enregistrée dans {self.config_file_path}")