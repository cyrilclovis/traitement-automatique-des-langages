from os import path

def pathExists(path_to_check: str):
    """Renvoie vrai si le chemin (fichier ou répertoire) correspondant existe"""
    return path.exists(path_to_check)
