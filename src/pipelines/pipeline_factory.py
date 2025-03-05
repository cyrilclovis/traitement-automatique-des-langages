######## Ce fichier regroupe les pipelines de deux types: Réutilisables et spécifiques
#### Comme ce fichier est assez long, voici une tables des matières des principales pipelines

# Faites CTrl + f et rechercher le nom pour voir la pipeline correspondante
    # tokenisation
    # truecase and cleaning
    # use openmt

from typing import List, Dict, TypedDict, Tuple

from src.pipelines.pipeline import Pipeline
from src.enums.command_enum import CommandType

# Import des classes factory
from src.commands.creation.factory.data_factory import DataGatheringCommandFactory
from src.commands.creation.factory.open_nmt_factory import OpenNMTCommandFactory
from src.commands.creation.factory.corpus_factory import CorpusConstructionCommandFactory
from src.commands.creation.factory.moses_factory import MosesCommandFactory 

from src.commands.config_command import ConfigCommand

# Types:
class CorpusSplits(TypedDict):
    train: str
    dev: str
    test: str

CorpusFilesByLang = Dict[str, CorpusSplits]  # Par exemple {"en": CorpusSplits, "fr": CorpusSplits}


class PipelineFactory:

    # Ce sont les factory depuis lesquels on récupère les commandes qui nous intéressent
    data_gathering_cmd_factory = DataGatheringCommandFactory()
    open_nmt_cmd_factory = OpenNMTCommandFactory()
    corpus_cmd_factory = CorpusConstructionCommandFactory()
    moses_cmd_factory = MosesCommandFactory()

    # ********************* Pipelines réutilisables
    # Ce sont des pipelines qui regroupe des opérations que l'on sera amené à répéter; moyennant quoi, parfois, de nombreux paramètres doivent etre passés !

    # tokenisation

    @staticmethod
    def tokenize_all_files(raw_corpus_splits: CorpusSplits, lang_code: str) -> Tuple[Pipeline, CorpusSplits]:
        """TODO"""

        pipeline = Pipeline()

        # Génére les noms des fichiers tokenisés
        tokensized_files_names: CorpusSplits = {
            split: insert_before_extension(raw_corpus_splits[split], ".tok", lang_code)
            for split in ["train", "dev", "test"]
        }

        # Ajoute les commandes au pipeline
        for split, output_file in tokensized_files_names.items():
            pipeline.add_command_from_factory(
                PipelineFactory.moses_cmd_factory,
                CommandType.TOKENIZE,
                input_file=raw_corpus_splits[split],
                output_file=output_file,
                lang=lang_code
            )

        return pipeline, tokensized_files_names

    @staticmethod
    def tokenize_all_corpora(raw_corpus_files_by_lang: CorpusFilesByLang, language_codes: List[str]) -> Tuple[Pipeline, CorpusFilesByLang]:

        """
        [PIPELINE I.2]
        /!\ Cette méthode travaille sur les les corpus tokensiés. Par exmeple, pour 'en', j'ai train.tok, test.tok, dev.tok
        """

        pipeline = Pipeline()

        tokenized_files_by_lang = {}

        for lang_code in language_codes:
            # Tokenisation des fichiers pour une langue
            tokenize_pipeline, tokenized_files_names = PipelineFactory.tokenize_all_files(raw_corpus_files_by_lang[lang_code], lang_code)
            pipeline.add_command(tokenize_pipeline)
            tokenized_files_by_lang[lang_code] = tokenized_files_names

        return pipeline, tokenized_files_by_lang

    # TODO

    @staticmethod
    def split_tokenized_corpus_into_train_dev_test(
        folder_base_path: str,
        file_name_prefix: str,
        code: str,
        nb_lines_for_train_corpus: int = 10000,
        nb_lines_for_dev_corpus: int = 1000,
        nb_lines_for_test_corpus: int = 500
    ) -> Tuple[Pipeline, dict]:
        """
        Cette pipeline permet de :
        1. Télécharger et extraire un fichier compressé contenant un texte tokenisé.
        2. Construire trois corpus distincts à partir de ce fichier: TRAIN, DEV, TEST

        Retourne :
        - La pipeline configurée.
        - Un dictionnaire contenant les chemins des fichiers de sortie.
        """

        train_file = f"{folder_base_path}/{file_name_prefix}_train_{format_k(nb_lines_for_train_corpus)}.tok.{code}"
        dev_file = f"{folder_base_path}/{file_name_prefix}_dev_{format_k(nb_lines_for_dev_corpus)}.tok.{code}"
        test_file = f"{folder_base_path}/{file_name_prefix}_test_{format_k(nb_lines_for_test_corpus)}.tok.{code}"

        pipeline = Pipeline().add_command_from_factory(
            # Téléchargement des données tokenisés en anglais (=> on a le fichier en.tok)
            PipelineFactory.data_gathering_cmd_factory,
            CommandType.DOWNLOAD_AND_EXTRACT_FROM_ZIP,
            url=f"https://object.pouta.csc.fi/OPUS-Europarl/v8/mono/{code}.tok.gz", dest_dir=folder_base_path
        ).add_command_from_factory(
            # ************************* On a obtenu deux fichiers tokenisés pour chacun d'entre eux, on crée les corpus TRAIN, DEV, TEST
            PipelineFactory.corpus_cmd_factory,
            CommandType.CORPUS_SPLITTING_INTO_TRAIN_DEV_TEST_CORPUSES,
            # ****** Commun à tous
            tokenized_corpus_path=f"{folder_base_path}/{code}.tok",

            # Corpus train
            nb_lines_to_extract_for_train_corpus=nb_lines_for_train_corpus,  # Nombre de lignes à extraire pour le corpus d'entraînement
            output_file_for_train_corpus=train_file,  # Fichier de sortie pour l'entraînement
            
            # Corpus dev
            starting_point_dev=nb_lines_for_train_corpus+1,
            nb_lines_to_extract_for_dev_corpus=nb_lines_for_dev_corpus,  # Nombre de lignes pour le corpus de validation
            output_file_for_dev_corpus=dev_file,  # Fichier de sortie pour la validation
            
            # Corpus test
            starting_point_test=nb_lines_for_train_corpus+nb_lines_for_dev_corpus+1,
            nb_lines_to_extract_for_test_corpus=nb_lines_for_test_corpus,  # Nombre de lignes pour le corpus de test
            output_file_for_test_corpus=test_file,  # Fichier de sortie pour le test
            exclude_file=dev_file  # Fichier à exclure (corpus dev) pour le test
        )
        
        return pipeline, {
            "train": train_file,
            "dev": dev_file,
            "test": test_file
        }
    
    # truecase and cleaning
    
    @staticmethod
    def train_and_apply_truecasing(tokenized_corpus_splits: CorpusSplits, lang_code: str, folder_base_path: str) -> Tuple[Pipeline, CorpusSplits]:
        """
        Entraîne le modèle Truecaser et l'applique sur les corpus TRAIN, DEV et TEST.

        Args:
            code: Code de la langue (ex: en, fr)
            model_path (str): Le chemin où sauvegarder le modèle de Truecasing.
            files_names (dict): Dictionnaire contenant les chemins des fichiers (train, dev, test).

        Returns:
            Pipeline: La pipeline mise à jour avec les étapes d'entraînement et d'application du Truecasing.
        """
        model_path = f"{folder_base_path}/truecase-model.{lang_code}"

        # *********** On entraine le modèle
        pipeline = Pipeline().add_command_from_factory(
            PipelineFactory.moses_cmd_factory,
            CommandType.TRAIN_TRUECASER_MODEL,
            model_path=model_path,
            corpus_path=tokenized_corpus_splits["train"]
        )

        # *********** Application du Truecasing sur TRAIN, DEV et TEST
        # Définition des noms de fichiers après le true casing
        truecased_files_names: CorpusSplits = {
            split: insert_before_extension(tokenized_corpus_splits[split], ".true", lang_code)
            for split in ["train", "dev", "test"]
        }

        # Ajout des commandes au pipeline
        for split, output_file in truecased_files_names.items():
            pipeline.add_command_from_factory(
                PipelineFactory.moses_cmd_factory,
                CommandType.TRUE_CASING,
                model_path=model_path,
                input_file=tokenized_corpus_splits[split],
                output_file=output_file
            )

        return pipeline, truecased_files_names

    @staticmethod
    def clean_and_truecase_all_corpus(truecased_files_by_lang: CorpusFilesByLang, language_codes: List[str], min_len=1, max_len=80) -> Tuple[Pipeline, CorpusSplits]:
        # ***** Application du nettoyage sur TRAIN, DEV et TEST pour le couple situé dans language_codes.
        pipeline = Pipeline()
        cleaned_files_names: CorpusSplits = {}
        lang_src = language_codes[0]

        for split in ["train", "dev", "test"]:

            input_file = remove_part_from_filename(truecased_files_by_lang[lang_src][split], f".{lang_src}") # On enlève le ".{lang_code}", ici .en ou .fr (on enlève toujours le dernier)
            output_file = f"{input_file}.clean"

            pipeline.add_command_from_factory(
                PipelineFactory.moses_cmd_factory,
                CommandType.CLEAN_CORPUS,
                input_file=input_file,
                lang1=language_codes[0],
                lang2=language_codes[1],
                output_file=output_file,
                min_len=min_len,
                max_len=max_len
            )

            # Enregistre le nom du fichier transformé dans le dictionnaire
            cleaned_files_names[split] = output_file
     
        return pipeline, cleaned_files_names
    
    @staticmethod
    def truecase_and_clean_corpora_pipeline(tokenized_corpus_files_by_lang: CorpusFilesByLang, language_codes: List[str], folder_base_path: str
                                           ) -> Tuple[Pipeline, CorpusSplits]:
        """
        [PIPELINE I.2]
        TODO
        """
        pipeline = Pipeline()
        truecased_files_by_lang: CorpusFilesByLang = {}

        for lang_code in language_codes:

            # Applique le Truecasing après avoir entraîné le modèle
            train_and_apply_casing_pipeline, truecased_files_names = PipelineFactory.train_and_apply_truecasing(tokenized_corpus_files_by_lang[lang_code], lang_code, folder_base_path)
            pipeline.add_command(train_and_apply_casing_pipeline)

            # Construire le dictionnaire CorpusFilesByLang pour chaque langue
            truecased_files_by_lang[lang_code] = truecased_files_names

        # ***** Application du nettoyage sur TRAIN, DEV et TEST (après avoir obtenu les corpus TRAIN, DEV, TEST nettoyés pour les langues contenus dans language_codes).
        clean_truecased_pipeline, clean_truecased_files_names = PipelineFactory.clean_and_truecase_all_corpus(truecased_files_by_lang, language_codes)
        pipeline.add_command(clean_truecased_pipeline)
     
        return pipeline, clean_truecased_files_names
    
    # use openmt
    
    @staticmethod
    def train_openmt_model_and_translate_pipeline(clean_truecased_files_names: CorpusSplits, language_codes: List[str], folder_base_path: str,  yaml_config_path: str, n_sample = 1000) -> Pipeline:
        """
        [PIPELINE I.2]
        """
        src_lang = language_codes[0]
        dest_lang = language_codes[1]
        folder_base_path += f"/{src_lang}_{language_codes[1]}"

        model_path = f"{folder_base_path}/run/model_step_{n_sample}.pt" # Chemin vers le modèle
        source_translation_file = f"{clean_truecased_files_names['test']}.{src_lang}" #  (Attention, c'est la source !) => A partir de cela, on traduit
        reference_translation_file = f"{clean_truecased_files_names['test']}.{dest_lang}" # (Attention, c'est la destination !) => On compare la traduction du modele avec ce fichier
        model_translation = f"{folder_base_path}/run/pred_{n_sample}.txt"

        config_command = ConfigCommand(folder_base_path, clean_truecased_files_names, language_codes, yaml_config_path)

        return Pipeline().add_command(
            # On crée le fichier YAML
            config_command
        ).add_command_from_factory(
            # On construit le vocabulaire
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.BUILD_VOCAB,
            config_path=yaml_config_path,
            src_vocab=config_command.get_vocab_source_path(),
            tgt_vocab=config_command.get_vocab_target_path(),
            n_sample=n_sample
        ).add_command_from_factory(
            # On fait l'entraiement
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRAIN,
            config_path=yaml_config_path,
            model_path = model_path,
        ).add_command_from_factory(
            # On traduit
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRANSLATE,
            model_path = model_path,
            src_path = source_translation_file,
            output_path = model_translation
        ).add_command_from_factory(
            # On évalue
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.BLEU_SCORE,
            reference_file=reference_translation_file,
            translation_file=model_translation
        )
        
        

    # ********************* Pipelines spécifiques

    @staticmethod
    def get_pipeline_i1(n_sample: int = 1000) -> Pipeline:
        """
        [PIPELINE I.1 => Expérimentation]
        """
        pipeline = Pipeline()

        pipeline.add_command_from_factory(
            # On télécharge les données depuis internet
            PipelineFactory.data_gathering_cmd_factory,
            CommandType.DOWNLOAD_AND_EXTRACT_FROM_TAR,
            url="https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz", dest_dir="./data"
        ).add_command_from_factory(
            # On crée le fichier YAML
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.YAML_CONFIG,
            config_path="./config/toy-ende.yaml"
        ).add_command_from_factory(
            # On construit le vocabulaire
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.BUILD_VOCAB,
            config_path="./config/toy-ende.yaml", n_sample=n_sample
        ).add_command_from_factory(
            # On fait l'entraiement
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRAIN,
            config_path="./config/toy-ende.yaml"
        ).add_command_from_factory(
            # On traduit
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRANSLATE,
            model_path = f"./data/toy-ende/run/model_step_{n_sample}.pt",
            src_path = "./data/toy-ende/src-test.txt",
            output_path = f"./data/toy-ende/run/pred_{n_sample}.txt"
        )
        return pipeline
    
    @staticmethod
    def get_pipeline_i2() -> Pipeline:
        """
        [PIPELINE I.1 => Expérimentation]
        """
        # 0) Ensemble des variables nécessaire pour l'execution de cette pipeline
        folder_base = "./data/provided-corpus/"
        language_codes=["en", "fr"] # Important source à gauche !
        yaml_config_path="./config/provided-corpus-europarl_en_fr.yaml"

        # I) On récupère les chemins vers les corpus donnés par le projet

        raw_corpus_files_by_lang: CorpusFilesByLang = {
            "en": {
                "train": folder_base + "Europarl_train_10k.en",
                "dev": folder_base + "Europarl_dev_1k.en",
                "test": folder_base + "Europarl_test_500.en"
            },
            "fr": {
                "train": folder_base + "Europarl_train_10k.fr",
                "dev": folder_base + "Europarl_dev_1k.fr",
                "test": folder_base + "Europarl_test_500.fr"
            }
        }

        # II) Tokenisation
        pipeline, tokenized_corpus_files_by_lang = PipelineFactory.tokenize_all_corpora(
            raw_corpus_files_by_lang=raw_corpus_files_by_lang,
            language_codes=language_codes)

        
        # II) On prépare les données (en appliquant le true casing et le nettoyage pour les 2 versions (langue source, langue cible)
        pipeline, clean_truecased_files_names = PipelineFactory.truecase_and_clean_corpora_pipeline(
            tokenized_corpus_files_by_lang=tokenized_corpus_files_by_lang,
            language_codes=language_codes,
            folder_base_path=folder_base
        )
        
        
        # III) On construit le vocabulaire, on entraine et on traduit
        pipeline.add_command(
            PipelineFactory.train_openmt_model_and_translate_pipeline(
            clean_truecased_files_names=clean_truecased_files_names,
            language_codes=language_codes,
            folder_base_path=folder_base,
            yaml_config_path=yaml_config_path
            )
        )

        return pipeline

            
    @staticmethod
    def standard_pipeline() -> Pipeline:

        """
        pipeline, clean_truecased_files_names = PipelineFactory.truecase_and_clean_corpora_pipeline(
            folder_base_path="./data/europarl",
            file_name_prefix="Europarl",
            language_codes=["en", "fr"]
        )


        cmd2 = PipelineFactory.train_openmt_model_and_translate_pipeline(
            folder_base_path="./data/europarl",
            clean_truecased_files_names=clean_truecased_files_names,
            language_codes=["en", "fr"],
            yaml_config_path="./config/europarl_en_fr.yaml"
        )

        return Pipeline().add_command(pipeline).add_command(cmd2)
        """
        pass
    


# ************************* UTLIS

# Fonction auxiliaire pour formater les chemins de fichiers
def format_k(x):
    return f"{x // 1000}k" if x >= 1000 else f"{x}"

# Permet d'insérer un suffix avant l'extension. Cela est très utile car le respect des noms de fichier
# est important pour l'utilisation des commandes de moses.
def insert_before_extension(file_name: str, suffix: str, extension: str) -> str:
    if extension not in file_name:
        raise ValueError(f"'{extension}' n'est pas dans '{file_name}'")
    return file_name.replace(f".{extension}", f"{suffix}.{extension}", 1)

def remove_part_from_filename(file_name: str, part_to_remove: str) -> str:
    if part_to_remove not in file_name:
        raise ValueError(f"'{part_to_remove}' n'est pas dans' '{file_name}'")
    return file_name.replace(f"{part_to_remove}", "", 1)

def add_part_in_filename(file_name: str, part_to_add: str):
    if part_to_add in file_name:
        raise ValueError(f"'{part_to_add}' est déjà dans' '{file_name}'")
    return file_name + part_to_add

