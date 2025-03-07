import yaml
import nltk
from typing import Dict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

# Téléchargement des ressources pour NLTK
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class LemmatizerCommand:
    def __init__(self, lang_code: str, input_file: str, output_file: str):
        """
        Initialise le lemmatizer avec la langue, les fichiers d'entrée et sortie, et le chemin de config.
        """
        self.lang_code = lang_code
        self.input_file = input_file
        self.output_file = output_file

        # Sélection du lemmatizer
        if lang_code == "en":
            self.lemmatizer = WordNetLemmatizer()
        elif lang_code == "fr":
            self.lemmatizer = FrenchLefffLemmatizer()
        else:
            raise ValueError("Langue non supportée. Utiliser 'fr' ou 'en'.")


    def get_wordnet_pos(self, word):
        """
        Important: WordNetLemmatizer a besoin d'étiquettes grammaticales (pos). Vrai pour l'anglais, on peut s'en passer
        en francais car la forme d'un mot donne souvent assez d'information pour retrouver son lemme (ex: chantais, chanté, chantons => chanter)
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)


    def lemmatize_text(self, text: str) -> str:
        """Lemmatisation d'un texte selon la langue spécifiée."""
        tokens = word_tokenize(text) if self.lang_code == "en" else text.split()
        if self.lang_code == "en":
            lemmatized_tokens = [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) for token in tokens]
        else:
            lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmatized_tokens)


    def lemmatize_corpus(self):
        """
        Applique la lemmatisation au fichier d'entrée et écrit le résultat dans le fichier de sortie.
        """
        with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                lemmatized_line = self.lemmatize_text(line.strip())
                outfile.write(lemmatized_line + '\n')


    def execute(self):
        """
        Exécute la lemmatisation et sauvegarde la configuration.
        """
        self.lemmatize_corpus()


if __name__ == "__main__":
    command = LemmatizerCommand(lang_code="fr", input_file="test_fr.txt", output_file="output_fr.txt")
    command.execute()