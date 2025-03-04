from typing import List, Dict, Optional


from src.pipelines.pipeline import Pipeline
from src.enums.command_enum import CommandType

# Import des classes factory
from src.commands.creation.factory.data_factory import DataGatheringCommandFactory
from src.commands.creation.factory.open_nmt_factory import OpenNMTCommandFactory
from src.commands.creation.factory.corpus_factory import CorpusConstructionCommandFactory
from src.commands.creation.factory.moses_factory import MosesCommandFactory 

from src.commands.config_command import ConfigCommand

class PipelineFactory:

    # Ce sont les factory depuis lesquels on récupère les commandes qui nous intéressent
    data_gathering_cmd_factory = DataGatheringCommandFactory()
    open_nmt_cmd_factory = OpenNMTCommandFactory()
    corpus_cmd_factory = CorpusConstructionCommandFactory()
    moses_cmd_factory = MosesCommandFactory()

    # ********************* Pipelines réutilisables
    # Ce sont des pipelines qui regroupe des opérations que l'on sera amené à répéter; moyennant quoi, parfois, de nombreux paramètres doivent etre passés !
    @staticmethod
    def tokenize_all_files(raw_corpus_files_names: Dict[str, str], lang_code: str) -> tuple[Pipeline, dict]:
        """TODO"""

        pipeline = Pipeline()

        tokensized_files_names = {}

        for split in ["train", "dev", "test"]:

            input_file = raw_corpus_files_names[split]
            output_file = insert_before_extension(input_file, ".tok", lang_code)

            pipeline.add_command_from_factory(
                PipelineFactory.moses_cmd_factory,
                CommandType.TOKENIZE,
                input_file=input_file,
                output_file=output_file,
                lang=lang_code
            )

            tokensized_files_names[split] = output_file

        return pipeline, tokensized_files_names
        
    @staticmethod
    def split_tokenized_corpus_into_train_dev_test(
        folder_base_path: str,
        file_name_prefix: str,
        code: str,
        nb_lines_for_train_corpus: int = 10000,
        nb_lines_for_dev_corpus: int = 1000,
        nb_lines_for_test_corpus: int = 500
    ) -> tuple[Pipeline, dict]:
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
    
    @staticmethod
    def train_and_apply_truecasing(folder_base_path: str, code: str, corpora_files_names: dict) -> tuple[Pipeline, dict]:
        """
        Entraîne le modèle Truecaser et l'applique sur les corpus TRAIN, DEV et TEST.

        Args:
            code: Code de la langue (ex: en, fr)
            model_path (str): Le chemin où sauvegarder le modèle de Truecasing.
            files_names (dict): Dictionnaire contenant les chemins des fichiers (train, dev, test).

        Returns:
            Pipeline: La pipeline mise à jour avec les étapes d'entraînement et d'application du Truecasing.
        """
        model_path = f"{folder_base_path}/truecase-model.{code}"

        # *********** On entraine le modèle
        pipeline = Pipeline().add_command_from_factory(
            PipelineFactory.moses_cmd_factory,
            CommandType.SOLVE_DEPENDENCIES_AND_TRAIN_TRUECASER_MODEL,
            model_path=model_path,
            corpus_path=corpora_files_names["train"]
        )

        # *********** Application du Truecasing sur TRAIN, DEV et TEST
        truecased_files_names = {}

        for split in ["train", "dev", "test"]:
            output_file = insert_before_extension(corpora_files_names[split], ".true", code)

            pipeline.add_command_from_factory(
                PipelineFactory.moses_cmd_factory,
                CommandType.TRUE_CASING,
                model_path=model_path,
                input_file=corpora_files_names[split],
                output_file=output_file
            )

            # Enregistre le nom du fichier transformé dans le dictionnaire
            truecased_files_names[split] = output_file

        return pipeline, truecased_files_names

    @staticmethod
    def clean_truecased_files(truecased_files_names: dict, language_codes: List[str], min_len=1, max_len=80) -> tuple[Pipeline, dict]:
        # ***** Application du nettoyage sur TRAIN, DEV et TEST pour le couple situé dans language_codes.
        pipeline = Pipeline()
        cleaned_files_names = {}

        for split in ["train", "dev", "test"]:

            input_file = remove_part_from_filename(truecased_files_names[split], f".{language_codes[1]}") # On enlève le ".{lang_code}", ici .en ou .fr (on enlève toujours le dernier)
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
    def build_and_clean_corpus_pipeline(folder_base_path: str,
                                        language_codes: List[str],
                                        file_name_prefix: str=None,
                                        raw_corpus_files: Optional[Dict[str, Dict[str, str]]] = None) -> tuple[Pipeline, dict]:
        """
        [PIPELINE I.2]
        """

        if file_name_prefix == None and raw_corpus_files == None:
            raise ValueError("Le corpus doit être construit avec file_name_prefix ou (exclusif) fourni via raw_corpus_files .")

        pipeline = Pipeline()

        for lang_code in language_codes:

            # ***** Pour chaque lang_code, on met en place les corpus, on entraine le modèle à partir du corpus train, puis on l'applique sur tous nos corpus.
            if file_name_prefix != None:
                build_corpora_pipeline, corpora_files_names = PipelineFactory.split_tokenized_corpus_into_train_dev_test(folder_base_path, file_name_prefix, lang_code)
                pipeline.add_command(build_corpora_pipeline)

            if raw_corpus_files:
                tokenize_pipeline, tokenized_files_names = PipelineFactory.tokenize_all_files(raw_corpus_files[lang_code], lang_code)
                pipeline.add_command(tokenize_pipeline)
                corpora_files_names = tokenized_files_names

            train_and_apply_casing_pipeline, truecased_files_names = PipelineFactory.train_and_apply_truecasing(folder_base_path, lang_code, corpora_files_names)
            pipeline.add_command(train_and_apply_casing_pipeline)

        # ***** Application du nettoyage sur TRAIN, DEV et TEST (après avoir obtenu les corpus TRAIN, DEV, TEST nettoyés pour les langues contenus dans language_codes).
        clean_truecased_pipeline, clean_truecased_files_names = PipelineFactory.clean_truecased_files(truecased_files_names, language_codes)
        pipeline.add_command(clean_truecased_pipeline)
     
        return pipeline, clean_truecased_files_names
    
    @staticmethod
    def train_openmt_model_and_translate_pipeline(folder_base_path: str, clean_truecased_files_names: dict, language_codes: List[str], yaml_config_path: str, n_sample = 1000) -> Pipeline:
        """
        [PIPELINE I.2]
        """
        folder_base_path += f"/{language_codes[0]}_{language_codes[1]}"

        model_path = f"{folder_base_path}/run/model_step_{n_sample}.pt" # Chemin vers le modèle
        src_path = f"{clean_truecased_files_names['test']}.{language_codes[0]}" # Chemin vers le fichier test true.cleaned (Attention, c'est la source !)
        output_path = f"{folder_base_path}/run/pred_{n_sample}.txt"

        return Pipeline().add_command(
            # On crée le fichier YAML
            ConfigCommand(folder_base_path, clean_truecased_files_names, language_codes, yaml_config_path)
        ).add_command_from_factory(
            # On construit le vocabulaire
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.BUILD_VOCAB,
            config_path=yaml_config_path, n_sample=n_sample
        ).add_command_from_factory(
            # On fait l'entraiement
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRAIN,
            config_path=yaml_config_path
        ).add_command_from_factory(
            # On traduit
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRANSLATE,
            model_path = model_path,
            src_path = src_path,
            output_path = output_path
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

        raw_corpus_files = {
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

        # II) On prépare les données (en appliquant le true casing et le nettoyage pour les versions anglaises et francaises)
        pipeline, clean_truecased_files_names = PipelineFactory.build_and_clean_corpus_pipeline(
            folder_base_path=folder_base,
            language_codes=language_codes,
            raw_corpus_files=raw_corpus_files
        )

        # III) On construit le vocabulaire, on entraine et on traduit
        pipeline.add_command(
            PipelineFactory.train_openmt_model_and_translate_pipeline(
            folder_base_path=folder_base,
            clean_truecased_files_names=clean_truecased_files_names,
            language_codes=language_codes,
            yaml_config_path=yaml_config_path
            )
        )

        return pipeline

            
    @staticmethod
    def standard_pipeline() -> Pipeline:

        
        pipeline, clean_truecased_files_names = PipelineFactory.build_and_clean_corpus_pipeline(
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

