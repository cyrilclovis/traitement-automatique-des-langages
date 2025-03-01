### Génère le fichier .yaml UNIQUEMENT POUR TOY-ENDE pour le moment. A terme, on souhaite utiliser le builder pattern avec cette classe
### pour construire un YAML sur-mesure !

import os
import yaml

class YAMLConfig:
    def __init__(self, folder_name, gpu=True,
                 save_checkpoint_steps=500, train_steps=None, valid_steps=None):
        """
        Initialise la configuration pour générer le fichier YAML.

        :param folder_name: Nom du dossier dans lequel on sauvegarde les données.
        :param gpu: Indique si le GPU doit être utilisé.
        :param save_checkpoint_steps: Nombre de pas entre les sauvegardes de checkpoints.
        :param train_steps: Nombre de pas d'entraînement avant d'arrêter.
        :param valid_steps: Nombre de pas entre les étapes de validation.
        """
        self.base_path = './data/' + folder_name + "/run/"
        self.gpu = gpu
        self.save_checkpoint_steps = save_checkpoint_steps
        self.train_steps = train_steps
        self.valid_steps = valid_steps


    def generate_config(self):
        """
        Génère la configuration complète pour le fichier YAML.
        
        Cette fonction appelle les autres sous-fonctions pour générer les 
        sections de configuration indépendantes, puis les assemble.

        :return: Dictionnaire complet de la configuration.
        """
        config = self.generate_model_config()
        config['data'] = self.generate_data_config()

        # Ajouter les paramètres d'entraînement si spécifiés
        if self.train_steps is not None:
            config['train_steps'] = self.train_steps

        # Ajouter les paramètres de validation si spécifiés
        if self.valid_steps is not None:
            config['valid_steps'] = self.valid_steps

        # Ajouter la configuration GPU si activée
        if self.gpu is not None:
           config['gpu_ranks'] = [0]

        return config


    def generate_model_config(self):
        """
        Génère la configuration du modèle (chemins de sauvegarde, etc.).

        :return: Dictionnaire contenant les configurations liées au modèle.
        """
        return {
            'save_data': self.join_paths(self.base_path, 'model', 'checkpoints'),
            'src_vocab': self.join_paths(self.base_path, 'vocab', 'src-vocab.txt'),
            'tgt_vocab': self.join_paths(self.base_path, 'vocab', 'tgt-vocab.txt'),
            'overwrite': False,
            'world_size': 1,
            'save_model': self.join_paths(self.base_path, 'model'),
            'save_checkpoint_steps': self.save_checkpoint_steps
        }


    def generate_data_config(self):
        """
        Génère la configuration des données pour l'entraînement et la validation.

        :return: Dictionnaire contenant les chemins des fichiers de données.
        """
        data_config = {}

        data_config_base_path = self.base_path.rstrip('run/')

        if self.train_steps is not None:
            data_config['corpus_1'] = {
                'path_src': self.join_paths(data_config_base_path, 'src-train.txt'),
                'path_tgt': self.join_paths(data_config_base_path, 'tgt-train.txt')
            }
        
        if self.valid_steps is not None:
            data_config['valid'] = {
                'path_src': self.join_paths(data_config_base_path, 'src-val.txt'),
                'path_tgt': self.join_paths(data_config_base_path, 'tgt-val.txt')
            }
        
        return data_config


    def generate_gpu_config(self):
        """
        Génère la configuration GPU si nécessaire.

        :return: Dictionnaire contenant la configuration pour le GPU.
        """
        if self.gpu:
            return {'gpu_ranks': [0]}
        return {}

    # ************************ Des méthodes utils

    def save_to_yaml(self, file_path):
        """
        Sauvegarde la configuration générée dans un fichier YAML.

        :param file_path: Chemin du fichier où la configuration sera sauvegardée.
        """
        config = self.generate_config()
        with open(file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
        print(f"Configuration sauvegardée dans : {file_path}")


    def join_paths(self, *args):
        """
        Joint les parties d'un chemin ensemble en utilisant os.path.join.
        Cela permet de simplifier les appels répétitifs à os.path.join.

        :param args: Parties du chemin à joindre.
        :return: Le chemin complet.
        """
        return os.path.join(*args)


# Exemple d'utilisation
config = YAMLConfig("toy-ende", gpu=False,
                    save_checkpoint_steps=500, train_steps=1000, valid_steps=500)
config.save_to_yaml(os.path.join("./config/", 'toy-ende.yaml'))
