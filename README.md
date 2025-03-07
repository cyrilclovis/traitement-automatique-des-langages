# traitement-automatique-des-langages

## Environnement d'exécution

- **OS** : Linux ou WSL
- **Version de Python testée** : Python 3.9.21
- **Configuration matérielle** : GPU 0, NVIDIA GeForce RTX 3050

ℹ️ Uniquement testé avec `"gpu_ranks": [0]`

## Quelques précisions

`NB` : Ce projet étant automatisé, le fichier `config.yaml` est généré dynamiquement par les scripts.  
Si vous souhaitez modifier la configuration, ne modifiez pas directement les fichiers `.yaml` dans `./config`, car ils seront écrasés à chaque exécution. 

### Modifications générales:  
1. Allez dans `./src/pipelines/pipeline_factory.py`.  
2. Utilisez `Ctrl + F` pour rechercher la fonction suivante :  

   ```python
   get_yaml_config()
   ```
3. Apportez-y les modifications souhaitées.

### Modifications spécifiques:  
1. Toujours dans `./src/pipelines/pipeline_factory.py`.
2. Utilisez `Ctrl + F` pour rechercher l'instance que vous souhaitez modifier: 

   ```python
   PipelineFactory.get_yaml_config
   ```

## Installation

1. Cloner le projet puis changer de répertoire
```bash
cd ./traitement-automatique-des-langages
```

2. Créer un environnement virtuel
```bash
python3 -m venv .venv
```

`NB` : 📢 On recommande très fortement d'utiliser: `python3.9 -m venv .venv`

3. Activer l'environnement virtuel
```bash
source .venv/bin/activate
```
3. Installer les dépendances
```bash
pip install -r requirements.txt
```

`NB` : Ne pas oublier d'activer l'envrionnement virtuel à chaque fois que vous travaillez sur le projet.

## Utilisation

### Prérequis

Si vous lancez le programme, vous allez probablement avoir une erreur disant que, `lefff-3.4-addition.mlex` ou `lefff-3.4.mlex` est introuvable.

1. Exécutez la commande suivante :
```bash
python ./src/utils/install_lefff.py
```

2. Si le programme échoue, alors faites-le manuellement :
- Téléchargez le dépôt GitHub du [FrenchLefffLemmatizer](https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer/tree/master/french_lefff_lemmatizer/data)

- Prendre le dossier `data` et mettez-le dans `.venv/lib/python{python_version}/site-packages/french_lefff_lemmatizer`


### Lancement

```bash
python3 main.py
```
