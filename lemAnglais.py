import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Téléchargement des ressources nécessaires
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

print("lemmatizer creer !")


def lemmatize_english(words):
    lemmatizer = WordNetLemmatizer()
    a = []
    tokens = word_tokenize(words)
    for token in tokens:
        lemmetized_word = lemmatizer.lemmatize(token)
        a.append(lemmetized_word)
    sentence = " ".join(a)
    return sentence


"""
==> MEME QUE DANS LE FICHIER POUR LE FRANCAIS
"""
def lemmatize_corpus(input_file, output_file, lemmatizer_func):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            lemmatized_line = lemmatizer_func(line.strip())  # Applique la fonction de lemmatisation sur chaque ligne
            outfile.write(lemmatized_line + '\n')



# Exemple pour l'anglais (Europarl)
lemmatize_corpus('data/provided-corpus/Europarl_test_500.en', 'data/created/Europarl_test_500.en', lemmatize_english)
 
