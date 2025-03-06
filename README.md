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


4. Telecharger le dossier suivant 
https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer/tree/master/french_lefff_lemmatizer/data
 et le dossier /data mettre dans .venv/lib/python3.9/site-packages/french_lefff_lemmatizer

## Utilisation

```bash
python3 main.py
```
