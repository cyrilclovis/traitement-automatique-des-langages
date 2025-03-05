from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer


# Spécifiez manuellement l'emplacement du fichier LEFFF
french_lemmatizer = FrenchLefffLemmatizer(LEFFF_FILE_STORAGE='/data/provided-corpus/Europarl_test_500.fr')



def lemmatize_french(sentence):
    words = sentence.split()  # Séparer la phrase en mots
    lemmatized_words = [french_lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def lemmatize_corpus(input_file, output_file, lemmatizer_func):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            lemmatized_line = lemmatizer_func(line.strip())  # Applique la fonction de lemmatisation sur chaque ligne
            outfile.write(lemmatized_line + '\n')

# Exemple pour le français (Europarl)
lemmatize_corpus('Europarl_train_100k.fr', 'Europarl_train_100k_lemmatized.fr', lemmatize_french)
