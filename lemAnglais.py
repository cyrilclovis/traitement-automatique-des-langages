import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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

def lemmatize_english(sentence):

    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemmatized_sentence = []
    
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:        
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence) 
    """
    words = nltk.word_tokenize(sentence)  # Tokenisation
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)
    """

def lemmatize_corpus(input_file, output_file, lemmatizer_func):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            lemmatized_line = lemmatizer_func(line.strip())  # Applique la fonction de lemmatisation sur chaque ligne
            outfile.write(lemmatized_line + '\n')

# Exemple pour l'anglais (Europarl)
lemmatize_corpus('data/provided-corpus/a.en', 'data/created/a.en', lemmatize_english)

print(lemmatizer.lemmatize("running"))
print(lemmatizer.lemmatize("better", pos="a"))  # Ajoute un "pos" pour spécifier que c'est un adjectif



# Tokenisation de la phrase
tokens = nltk.word_tokenize("studies studying cries cry.")

# Application de l'étiquetage des parties du discours
tags = nltk.pos_tag(tokens)
print(tags)

print(lemmatize_english("studies studying cries cry."))
