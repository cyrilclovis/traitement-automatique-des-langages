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

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ  # Adjectif
    elif nltk_tag.startswith('V'):
        return wordnet.VERB  # Verbe
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN  # Nom
    elif nltk_tag.startswith('R'):
        return wordnet.ADV  # Adverbe
    else:          
        return None


def lemmatize_english(words):
    lemmatizer = WordNetLemmatizer()
    a = []
    tokens = word_tokenize(words)
    for token in tokens:
        lemmetized_word = lemmatizer.lemmatize(token)
        a.append(lemmetized_word)
    sentence = " ".join(a)
    return sentence



def lemmatize_corpus(input_file, output_file, lemmatizer_func):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            lemmatized_line = lemmatizer_func(line.strip())  # Applique la fonction de lemmatisation sur chaque ligne
            outfile.write(lemmatized_line + '\n')

# Exemple pour l'anglais (Europarl)
lemmatize_corpus('data/provided-corpus/Europarl_test_500.en', 'data/created/Europarl_test_500.en', lemmatize_english)

print(lemmatizer.lemmatize("running"))
print(lemmatizer.lemmatize("better", pos="a"))  # Ajoute un "pos" pour spécifier que c'est un adjectif



# Tokenisation de la phrase
tokens = nltk.word_tokenize("studies studying cries cry.")

# Application de l'étiquetage des parties du discours
tags = nltk.pos_tag(tokens)
print(tags)

print(lemmatize_english("studies studying cries cry."))
