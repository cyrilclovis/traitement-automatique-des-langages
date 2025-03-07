# traitement-automatique-des-langages

## Installation

1. Cloner le projet puis changer de répertoire
```bash
cd ./traitement-automatique-des-langages
```

2. Créer un environnement virtuel
```bash
python3 -m venv .venv
```

`NB` : Si vous le souhaitez, vous pouvez être plus précis concernant la version de Python à utiliser. Par exemple: `python3.9 -m venv .venv`

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

⚠️ Remarque : Cette erreur a été observée sous **Python 3.9**.

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
