import yaml
from typing import Dict, List, Any

class ConfigCommand:
    allowed_fields = {
        # vocab
        "src_vocab": str, "tgt_vocab": str,
        # save param 
        "save_model": str, "save_data": str, "overwrite": bool,
        # Training param
        "save_checkpoint_steps": int, "train_steps": int, "valid_steps": int,
        # Training conditions (depends on user's PC configuration)
        "world_size": int, "gpu_ranks": list,
        # data
        "data": dict
    }


    def __init__(self, config_file_path: str, run_base_path:str):
        self.config_file_path = config_file_path
        self.run_base_path = run_base_path
        self.config = {
            "src_vocab": "", "tgt_vocab": "",
            "overwrite": False,
            "save_model": "", "save_data": "",
            "save_checkpoint_steps": 500,
            "train_steps": 1000, "valid_steps": 500,
            "world_size": 1, "gpu_ranks": [0],
            "data": {}
        }


    # ************ Getters
    def get_config_file_path(self):
        """Renvoie le chemin vers le fichier de configuration .yaml"""
        return self.config_file_path

    def get_run_base_path(self):
        """Renvoie le chemin vers le dossier run (Il contient tout ce qui concerne l'IA)"""
        return self.run_base_path

    def get(self, key: str) -> Any:
        """Retourne la valeur d'une clé, supporte les clés imbriquées."""
        keys = key.split(".")
        if keys[0] not in self.allowed_fields:
            raise KeyError(f"❌ Clé '{keys[0]}' non autorisée")
        
        if len(keys) == 1:
            return self.config[keys[0]]
        else:
            return self._get_nested(self.config, keys)


    def _get_nested(self, d: dict, keys: List[str]) -> Any:
        """Récupère une valeur dans une structure imbriquée."""
        for key in keys:
            d = d.get(key)
            if d is None:
                raise KeyError(f"❌ Clé '{'.'.join(keys)}' introuvable")
        return d
    

    # ************ Setters

    def set(self, values: Dict[str, Any]):
        """Ajoute ou met à jour des champs si autorisés."""
        for key, value in values.items():
            keys = key.split(".")
            if keys[0] not in self.allowed_fields:
                raise KeyError(f"❌ Clé '{keys[0]}' non autorisée")
            
            if len(keys) == 1:
                expected_type = self.allowed_fields[keys[0]]
                if not isinstance(value, expected_type):
                    raise TypeError(f"❌ Type invalide pour '{key}'. Attendu: {expected_type.__name__}")
                self.config[keys[0]] = value
            else:
                self._set_nested(self.config, keys, value)
        return self
    

    def _set_nested(self, d: dict, keys: List[str], value: Any):
        """Ajoute/modifie une valeur dans une structure imbriquée."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value


    def remove(self, key: str):
        """Supprime une clé de la configuration, supporte les clés imbriquées."""
        keys = key.split(".")
        if keys[0] not in self.allowed_fields:
            raise KeyError(f"❌ Clé '{keys[0]}' non autorisée")
        
        if len(keys) == 1:
            if keys[0] in self.config:
                del self.config[keys[0]]
                print(f"✅ Clé '{key}' supprimée.")
                return self
            else:
                raise KeyError(f"❌ Clé '{key}' introuvable")
        else:
            self._remove_nested(self.config, keys)
    

    def _remove_nested(self, d: dict, keys: List[str]):
        """Supprime une clé dans une structure imbriquée."""
        for key in keys[:-1]:
            d = d.get(key)
            if d is None:
                raise KeyError(f"❌ Clé '{'.'.join(keys)}' introuvable")
        if keys[-1] in d:
            del d[keys[-1]]
            print(f"✅ Clé '{'.'.join(keys)}' supprimée.")
        else:
            raise KeyError(f"❌ Clé '{'.'.join(keys)}' introuvable")


    def execute(self):
        with open(self.config_file_path, "w") as file:
            yaml.dump(self.config, file, default_flow_style=False)
        print(f"✅ Configuration enregistrée dans {self.config_file_path}")