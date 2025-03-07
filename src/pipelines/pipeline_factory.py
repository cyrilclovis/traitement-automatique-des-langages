######## Ce fichier regroupe les pipelines de deux types: Réutilisables et spécifiques
#### Comme ce fichier est assez long, voici une tables des matières des principales pipelines

# Faites CTrl + f et rechercher le nom pour voir la pipeline correspondante
    # split file into TRAIN, DEV, TEST corpora
    # tokenisation
    # lemmatizer
    # truecase and cleaning
    # use openmt

from typing import List, Dict, TypedDict, Tuple, Union

from src.pipelines.pipeline import Pipeline
from src.enums.command_enum import CommandType

# Import des classes factory
from src.commands.creation.factory.data_factory import DataGatheringCommandFactory
from src.commands.creation.factory.open_nmt_factory import OpenNMTCommandFactory
from src.commands.creation.factory.corpus_factory import CorpusConstructionCommandFactory
from src.commands.creation.factory.moses_factory import MosesCommandFactory 

from src.commands.config_command import ConfigCommand
from src.commands.lemmatizer_command import LemmatizerCommand


# Types:
class CorpusSplits(TypedDict):
    train: str
    dev: str
    test: str

CorpusFilesByLang = Dict[str, CorpusSplits]  # Par exemple {"en": CorpusSplits, "fr": CorpusSplits}

class ModelPathsInfo(TypedDict):
    """
    Structure pour stocker les chemins relatifs à un modèle de traduction.
    """
    folder_base_path: str  # Dossier dans lequel on trouve le modèle entrainé (ou dans une de ses sous-hierarchie)
    model_path: str        # Chemin vers le modèle entrainé 
    src_lang: str
    dest_lang: str

class PipelineFactory:

    # Ce sont les factory depuis lesquels on récupère les commandes qui nous intéressent
    data_gathering_cmd_factory = DataGatheringCommandFactory()
    open_nmt_cmd_factory = OpenNMTCommandFactory()
    corpus_cmd_factory = CorpusConstructionCommandFactory()
    moses_cmd_factory = MosesCommandFactory()

    # ********************* Pipelines réutilisables
    # Ce sont des pipelines qui regroupe des opérations que l'on sera amené à répéter; moyennant quoi, parfois, de nombreux paramètres doivent etre passés !

    # split file into TRAIN, DEV, TEST corpora

    @staticmethod
    def split_corpus_into_train_dev_test(
        # fichier à partir du quel, on crée les corpus
        corpus_file_path: str,
        # Pour la création des noms de fichiers
        folder_base_path: str,
        file_name_prefix: str,
        code: str,
        # Pour la découpe
        nb_lines_for_train_corpus: int,
        nb_lines_for_dev_corpus: int,
        nb_lines_for_test_corpus: int
    ) -> Tuple[Pipeline, CorpusSplits]:
        """
        Cette pipeline permet de :
        1. Télécharger et extraire un fichier compressé contenant un texte tokenisé.
        2. Construire trois corpus distincts à partir de ce fichier: TRAIN, DEV, TEST

        Retourne :
        - La pipeline configurée.
        - Un dictionnaire contenant les chemins des fichiers de sortie.
        """

        train_file = f"{folder_base_path}/{file_name_prefix}_train_{format_k(nb_lines_for_train_corpus)}.{code}"
        dev_file = f"{folder_base_path}/{file_name_prefix}_dev_{format_k(nb_lines_for_dev_corpus)}.{code}"
        test_file = f"{folder_base_path}/{file_name_prefix}_test_{format_k(nb_lines_for_test_corpus)}.{code}"

        pipeline = Pipeline().add_command_from_factory(
            # ************************* A partir du "gros" ficher, on crée les corpus TRAIN, DEV, TEST
            PipelineFactory.corpus_cmd_factory,
            CommandType.CORPUS_SPLITTING_INTO_TRAIN_DEV_TEST_CORPUSES,
            # ****** Commun à tous
            corpus_path=corpus_file_path,

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
    def build_all_corpora_by_splitting(
        # fichier à partir du quel, on crée les corpus
        corpus_file_path: str,
        language_codes: List[str],
        # Pour la création des noms de fichiers
        folder_base_path: str,
        file_name_prefix: str,
        # Pour la découpe
        nb_lines_for_train_corpus: int = 10000,
        nb_lines_for_dev_corpus: int = 1000,
        nb_lines_for_test_corpus: int = 500
    ) -> Tuple[Pipeline, CorpusFilesByLang]:

        """
        [PIPELINE I.2]
        /!\ Cette méthode travaille sur les les corpus tokensiés. Par exmeple, pour 'en', j'ai train.tok, test.tok, dev.tok
        """

        pipeline = Pipeline()

        corpus_files_by_lang = {}

        for lang_code in language_codes:
            # Construit des corpus TRAIN, DEV, TEST
            split_file_into_corpora_pipeline, split_files_names = PipelineFactory.split_corpus_into_train_dev_test(
                corpus_file_path=f"{corpus_file_path}.{lang_code}",
                folder_base_path=folder_base_path,
                file_name_prefix=file_name_prefix,
                code=lang_code,
                nb_lines_for_train_corpus=nb_lines_for_train_corpus,
                nb_lines_for_dev_corpus=nb_lines_for_dev_corpus,
                nb_lines_for_test_corpus=nb_lines_for_test_corpus
            )
            pipeline.add_command(split_file_into_corpora_pipeline)
            corpus_files_by_lang[lang_code] = split_files_names

        return pipeline, corpus_files_by_lang
    
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
    def tokenize_all_corpora(raw_corpus_files_by_lang: CorpusFilesByLang, language_codes: List[str], useLemmatizer:bool=False) -> Tuple[Pipeline, CorpusFilesByLang]:

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

            # Lemmatize les fichiers pour une langue
            if useLemmatizer:
                lemmatize_pipeline, lemmatized_files_names = PipelineFactory.lemmatize_all_files(tokenized_files_by_lang[lang_code], lang_code)
                pipeline.add_command(lemmatize_pipeline)
                tokenized_files_by_lang[lang_code] = lemmatized_files_names

        return pipeline, tokenized_files_by_lang
    
    # lemmatizer

    @staticmethod
    def lemmatize_all_files(tokenized_corpus_splits: CorpusSplits, lang_code: str) -> Tuple[Pipeline, CorpusSplits]:
        
        pipeline = Pipeline()

        # Génére les noms des fichiers tokenisés
        lemmatized_files_names: CorpusSplits = {
            split: insert_before_extension(tokenized_corpus_splits[split], ".lem", lang_code)
            for split in ["train", "dev", "test"]
        }

        # Ajoute les commandes au pipeline
        for split, output_file in lemmatized_files_names.items():
            pipeline.add_command(
                LemmatizerCommand(
                    lang_code=lang_code,
                    input_file=tokenized_corpus_splits[split],
                    output_file=output_file
                )
            )

        return pipeline, lemmatized_files_names

    # truecase and cleaning
    
    @staticmethod
    def train_and_apply_truecasing(tokenized_corpus_splits: CorpusSplits, lang_code: str, folder_base_path: str, current_corpus_name:str=None) -> Tuple[Pipeline, CorpusSplits]:
        """
        Entraîne le modèle Truecaser et l'applique sur les corpus TRAIN, DEV et TEST.

        Args:
            code: Code de la langue (ex: en, fr)
            model_path (str): Le chemin où sauvegarder le modèle de Truecasing.
            files_names (dict): Dictionnaire contenant les chemins des fichiers (train, dev, test).

        Returns:
            Pipeline: La pipeline mise à jour avec les étapes d'entraînement et d'application du Truecasing.
        """
        model_path = f"{folder_base_path}"
        if current_corpus_name:
            model_path += f"/{current_corpus_name.lower()}"
        model_path += f"/truecase-model.{lang_code}"

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
    def truecase_and_clean_corpora_pipeline(tokenized_corpus_files_by_lang: CorpusFilesByLang, language_codes: List[str], folder_base_path: str, current_corpus_name:str=None
                                           ) -> Tuple[Pipeline, CorpusSplits]:
        """
        [PIPELINE I.2]
        TODO
        """
        pipeline = Pipeline()
        truecased_files_by_lang: CorpusFilesByLang = {}

        for lang_code in language_codes:

            # Applique le Truecasing après avoir entraîné le modèle
            train_and_apply_casing_pipeline, truecased_files_names = PipelineFactory.train_and_apply_truecasing(tokenized_corpus_files_by_lang[lang_code], lang_code, folder_base_path, current_corpus_name)
            pipeline.add_command(train_and_apply_casing_pipeline)

            # Construire le dictionnaire CorpusFilesByLang pour chaque langue
            truecased_files_by_lang[lang_code] = truecased_files_names

        # ***** Application du nettoyage sur TRAIN, DEV et TEST (après avoir obtenu les corpus TRAIN, DEV, TEST nettoyés pour les langues contenus dans language_codes).
        clean_truecased_pipeline, clean_truecased_files_names = PipelineFactory.clean_and_truecase_all_corpus(truecased_files_by_lang, language_codes)
        pipeline.add_command(clean_truecased_pipeline)
     
        return pipeline, clean_truecased_files_names
    
    # use openmt
    # └─> YAML
    @staticmethod
    def get_yaml_config(yaml_config_path: str, clean_truecased_files_names: CorpusSplits, language_codes: List[str], folder_base_path: str, run_number:int, train_steps: int, valid_steps:int, save_checkpoint_steps:int, gpu:bool = True) -> ConfigCommand:
        """Renvoie une commande capable de créeer un fichier yaml utilisé lors de la construction du vocabulaire et de l'entrainement.
        On renvoie une commande que l'utilisateur peut mettre à jour (avec les fonctions set ou remove). Voyez cela comme un modèle de base"""

        src_lang=language_codes[0]
        dest_lang=language_codes[1]
        run_base_path = f"{folder_base_path}/{src_lang}_{dest_lang}/run{run_number}"

        config_command = ConfigCommand(yaml_config_path, run_base_path).set({
            # vocab
            "src_vocab": f"{run_base_path}/vocab.src",
            "tgt_vocab": f"{run_base_path}/vocab.tgt",
            # save param
            "save_model": f"{run_base_path}/model",
            "save_data": f"{run_base_path}/samples",
            "overwrite": False,
            # Training param
            "train_steps": train_steps,
            "valid_steps": valid_steps,
            "save_checkpoint_steps": save_checkpoint_steps,
            # Training conditions (depends on user's PC configuration)
            "world_size": 1,
            "gpu_ranks": [0],
            # data
            # └─> corpus_1
            "data.corpus_1.path_src": f"{clean_truecased_files_names['train']}.{src_lang}",
            "data.corpus_1.path_tgt": f"{clean_truecased_files_names['train']}.{dest_lang}",
            # └─> valid
            "data.valid.path_src": f"{clean_truecased_files_names['dev']}.{src_lang}",
            "data.valid.path_tgt": f"{clean_truecased_files_names['dev']}.{dest_lang}",
        })

        if gpu == False:
            config_command.remove("gpu_ranks")

        return config_command

    # └─> build vocab and train model
    @staticmethod
    def build_vocab_and_train_model(yaml_config: ConfigCommand, language_codes: List[str], folder_base_path: str, n_sample:int=-1) -> Tuple[Pipeline, ModelPathsInfo]:
        """
        [PIPELINE I.2 - BUILD_VOCAB & TRAIN]
        """
        src_lang = language_codes[0]
        dest_lang = language_codes[1]

        yaml_config_path = yaml_config.get_config_file_path()
        folder_base_path = yaml_config.get_run_base_path()
        model_path = f"{yaml_config.get_run_base_path()}/model_step_{yaml_config.get('train_steps')}.pt"  # Chemin vers le modèle

        pipeline = Pipeline()

        pipeline.add_command_from_factory(
            # On construit le vocabulaire
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.BUILD_VOCAB,
            config_path=yaml_config_path,
            src_vocab=yaml_config.get("src_vocab"),
            tgt_vocab=yaml_config.get("tgt_vocab"),
            **({"n_sample": n_sample} if n_sample is not None else {})
        ).add_command_from_factory(
            # On fait l'entraiement
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRAIN,
            config_path=yaml_config_path,
            model_path = model_path,
        )

        return pipeline, {
            "folder_base_path": folder_base_path,
            "model_path": model_path, # Chemin vers le modèle
            "src_lang": src_lang,
            "dest_lang": dest_lang
        }
        
    # └─> translation and evaluation
    @staticmethod
    def translate_and_evaluate(clean_truecased_files_names: CorpusSplits, model_paths_info: ModelPathsInfo, n_sample:int=-1) -> Pipeline:
        """
        [PIPELINE I.2 - TRANSLATE & BLEU_SCORE]
        """

        source_translation_file = f"{clean_truecased_files_names['test']}.{model_paths_info['src_lang']}"        # (Attention, c'est la source !) => A partir de cela, on traduit
        reference_translation_file = f"{clean_truecased_files_names['test']}.{model_paths_info['dest_lang']}"    # (Attention, c'est la destination !) => On compare la traduction du modele avec ce fichier
        model_translation = f"{model_paths_info['folder_base_path']}/pred_{n_sample if n_sample is not None else 'all'}.txt"

        pipeline = Pipeline()

        pipeline.add_command_from_factory(
            # On traduit
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.TRANSLATE,
            model_path = model_paths_info["model_path"],
            src_path = source_translation_file,
            output_path = model_translation
        ).add_command_from_factory(
            # On évalue
            PipelineFactory.open_nmt_cmd_factory,
            CommandType.BLEU_SCORE,
            reference_file=reference_translation_file,
            translation_file=model_translation
        )

        return pipeline
    
    @staticmethod
    def build_train_translate_evaluate_pipeline(yaml_config: ConfigCommand, clean_truecased_files_names: CorpusSplits, language_codes: List[str], folder_base_path: str, n_sample:int=-1, get_model_paths_info: bool = False) -> Union[Pipeline, Tuple[Pipeline, ModelPathsInfo]]:
        """
        [PIPELINE I.2]
        """
        pipeline = Pipeline()

        # Constructin du vocabulaire et entrainement
        build_and_train_pipeline, models_paths_info = PipelineFactory.build_vocab_and_train_model(
            yaml_config=yaml_config,
            language_codes=language_codes,
            folder_base_path=folder_base_path,
            **({"n_sample": n_sample} if n_sample is not None else {})

        )
        pipeline.add_command(build_and_train_pipeline)
        
        # Traduction et évaluation
        pipeline.add_command(
            PipelineFactory.translate_and_evaluate(
                clean_truecased_files_names=clean_truecased_files_names,
                model_paths_info=models_paths_info,
                **({"n_sample": n_sample} if n_sample is not None else {})
            )
        )

        if get_model_paths_info:
            return pipeline, models_paths_info
        return pipeline
    
    
    # ********************* Pipelines spécifiques
    
    @staticmethod
    def get_pipeline_i1() -> Pipeline:
        """
        [PIPELINE I.1 => Expérimentation]
        """
        pipeline = Pipeline()

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
        tokenize_pipeline, tokenized_corpus_files_by_lang = PipelineFactory.tokenize_all_corpora(
            raw_corpus_files_by_lang=raw_corpus_files_by_lang,
            language_codes=language_codes)
        pipeline.add_command(tokenize_pipeline)

        # II) On prépare les données (en appliquant le true casing et le nettoyage pour les 2 versions (langue source, langue cible)
        true_case_clean_pipeline, clean_truecased_files_names = PipelineFactory.truecase_and_clean_corpora_pipeline(
            tokenized_corpus_files_by_lang=tokenized_corpus_files_by_lang,
            language_codes=language_codes,
            folder_base_path=folder_base
        )
        pipeline.add_command(true_case_clean_pipeline)
        
        
        # III) On construit le vocabulaire, on entraine et on traduit
        # Pour YAML
        config_command = PipelineFactory.get_yaml_config(
            yaml_config_path=yaml_config_path,
            clean_truecased_files_names=clean_truecased_files_names,
            language_codes=language_codes,
            folder_base_path=folder_base,
            run_number= 1, # Pour l'exercice 1, il y a qu'un seul run
            train_steps= 10_000,
            valid_steps= 1_000,
            save_checkpoint_steps= 1_000,
        )

        pipeline.add_command(config_command).add_command(
            PipelineFactory.build_train_translate_evaluate_pipeline(
                yaml_config=config_command,
                clean_truecased_files_names=clean_truecased_files_names,
                language_codes=language_codes,
                folder_base_path=folder_base,
            )
        )

        return pipeline

            
    @staticmethod
    def get_pipeline_i2_run1(folder_base: str, language_codes:List[str], yaml_config_path_run1:str, useLemmatizer:bool) -> Tuple[Pipeline, dict[str, dict], dict]:
        """language_codes=["en", "fr"] # Important source à gauche !"""
        
        pipeline = Pipeline()

        # 0) Ensemble des variables nécessaire pour l'execution de cette pipeline
        yaml_config_path=yaml_config_path_run1
        lang_tuple = f"{language_codes[0]}-{language_codes[1]}"

        sources = {
            "Europarl": {
                "folder": f"{folder_base}/europarl",
                "url": f"https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/{lang_tuple}.txt.zip",
                "first_file": f"Europarl.{lang_tuple}.en",
                "second_file": f"Europarl.{lang_tuple}.fr",
                "prefix": "Europarl",
                "nb_lines_for_train_corpus":100_000,
                "nb_lines_for_dev_corpus":3750,
                "nb_lines_for_test_corpus":500
            },
            "EMEA": {
                "folder": f"{folder_base}/emea",
                "url": f"https://object.pouta.csc.fi/OPUS-EMEA/v3/moses/{lang_tuple}.txt.zip",
                "first_file": f"EMEA.{lang_tuple}.en",
                "second_file": f"EMEA.{lang_tuple}.fr",
                "prefix": "EMEA",
                "nb_lines_for_train_corpus":10_000,
                "nb_lines_for_dev_corpus":0,
                "nb_lines_for_test_corpus":500
            }
        }
        
        # I) Récupérer les fichiers depuis le web, puis crée les corpus TRAIN, DEV et TEST
        clean_truecased_files_names_by_source = {}
        
        for source_name, params in sources.items():
            first_file_path = f"{params['folder']}/{params['first_file']}"
        
            pipeline.add_command_from_factory(
                PipelineFactory.data_gathering_cmd_factory,
                CommandType.DOWNLOAD_AND_EXTRACT_FROM_ZIP,
                url=params["url"],
                dest_dir=params["folder"],
                first_file_to_extract=params["first_file"],
                second_file_to_extract=params["second_file"],
                output_path=first_file_path # utilisé pour vérifier si au moins un des deux fichiers a déjà été téléchargé (et donc ne pas re-télécharger)
            )
            
            raw_corpus_files_pipeline, raw_corpus_files_by_lang = PipelineFactory.build_all_corpora_by_splitting(
                corpus_file_path=first_file_path[:-3], # Enlève l'extension ".en", exemple on garde ./../../Europarl.en-fr, la pipeline ajoute déjà .lang_code
                language_codes=language_codes,
                folder_base_path=params["folder"],
                file_name_prefix=params["prefix"],
                nb_lines_for_train_corpus=params["nb_lines_for_train_corpus"],
                nb_lines_for_dev_corpus=params["nb_lines_for_dev_corpus"],
                nb_lines_for_test_corpus=params["nb_lines_for_test_corpus"],
            )
            pipeline.add_command(raw_corpus_files_pipeline)

            # II) Tokenisation
            tokenize_pipeline, tokenized_corpus_files_by_lang = PipelineFactory.tokenize_all_corpora(
                raw_corpus_files_by_lang=raw_corpus_files_by_lang,
                language_codes=language_codes,
                useLemmatizer=useLemmatizer
                )
            pipeline.add_command(tokenize_pipeline)

            
            # III) On prépare les données (en appliquant le true casing et le nettoyage pour les 2 versions (langue source, langue cible)
            true_case_clean_pipeline, clean_truecased_files_names = PipelineFactory.truecase_and_clean_corpora_pipeline(
                tokenized_corpus_files_by_lang=tokenized_corpus_files_by_lang,
                language_codes=language_codes,
                folder_base_path=folder_base,
                current_corpus_name=source_name
            )
            pipeline.add_command(true_case_clean_pipeline)

            # Ajout des fichiers nettoyés et truecasés dans le dictionnaire avec source_name comme clé
            clean_truecased_files_names_by_source[source_name] = clean_truecased_files_names
        
        # IV) On construit le vocabulaire, on entraine et on traduit [UNIQUEMENT POUR EUROPARL]
        # En effet, on rappelle que toute la partie EMEA existe UNIQUEMENT pour
        # 1) [RUN 1 et 2] Proposer le corpus test, pour faire la deuxième évaluation en hors-domaine
        # 2) [RUN 2] Ajouter 10k de ligne au 100k de Europarl

        ######################################## RUN 1

        # Pour YAML
        config_command = PipelineFactory.get_yaml_config(
            yaml_config_path=yaml_config_path,
            clean_truecased_files_names=clean_truecased_files_names_by_source["Europarl"],
            language_codes=language_codes,
            folder_base_path=folder_base,
            run_number=1,
            train_steps= 10_000,
            valid_steps= 1_000,
            save_checkpoint_steps= 1_000,
        )

        pipeline.add_command(config_command)
        
        train_translate_evaluate_pipeline, model_paths_info = PipelineFactory.build_train_translate_evaluate_pipeline(
            yaml_config=config_command,
            clean_truecased_files_names=clean_truecased_files_names_by_source["Europarl"],
            language_codes=language_codes,
            folder_base_path=folder_base,
            get_model_paths_info = True
        )

        # Evaluation:
        pipeline.add_command(
            # dont évaluation 1.1 => c'est à dire avec le domaine "classique"
            train_translate_evaluate_pipeline
        ).add_command(
            # évaluation 1.2 => c'est à dire hors domaine (on utilise EMEA)
            PipelineFactory.translate_and_evaluate(
                clean_truecased_files_names=clean_truecased_files_names_by_source["EMEA"],
                model_paths_info=model_paths_info,
            )
        )

        return pipeline, sources, clean_truecased_files_names_by_source

    @staticmethod
    def get_i2_all_run(folder_base_path: str, language_codes:List[str], yaml_config_path_run1:str, yaml_config_path_run2:str, useLemmatizer:bool=False) -> Pipeline:
        """Met en place les runs 1 et 2 de la partie II. Le run1 et run2, s'appuie sur les memes corpus (un détail, pour run2, on ajoute les 10k
        de ligne de train EMEA au 100k de train Europarl)
        """
        pipeline = Pipeline()

        # ************************************** Run1
        pipeline_run1, sources, clean_truecased_files_names_by_source_after_run1 = PipelineFactory.get_pipeline_i2_run1(
            folder_base=folder_base_path,
            language_codes=language_codes,
            yaml_config_path_run1=yaml_config_path_run1,
            useLemmatizer=useLemmatizer
        )
        pipeline.add_command(pipeline_run1)

        # ************************************** Run2
        # On ajoute le contenu du fichier TRAIN EMEA et Europarl dans un nouveau fichier (pour les version en et fr)
        nb_lines_train_europarl = sources["Europarl"]["nb_lines_for_train_corpus"]
        output_nb_lines = nb_lines_train_europarl + sources["EMEA"]["nb_lines_for_train_corpus"]
        output_file = clean_truecased_files_names_by_source_after_run1["Europarl"]["train"].replace(format_k(nb_lines_train_europarl), format_k(output_nb_lines))

        language_codes=["en", "fr"]

        for lang_code in language_codes:
            pipeline.add_command_from_factory(
                PipelineFactory.corpus_cmd_factory,
                CommandType.MERGE_FILES_INTO_NEW_FILE,
                src_file_1=f"{clean_truecased_files_names_by_source_after_run1['Europarl']['train']}.{lang_code}",
                src_file_2=f"{clean_truecased_files_names_by_source_after_run1['EMEA']['train']}.{lang_code}",
                output_file=f"{output_file}.{lang_code}", # On ajoute l'extension
            )

        ### --- NB: Lorsque l'on arrive, ici, tous les fichiers sont parfaitement "propres" (tokenisés, true cased, < 80 caractères)
        ### C'est aussi le cas pour le fichier tout juste créer (EMEA + Europarl), car construit à partir de deux fichiers "propres"

        # Il reste donc toute la partie openmt (exactement comme dans get_pipeline_i2)

        clean_truecased_files_names_by_source_after_run1["Europarl"]["train"] = output_file

        # Pour YAML
        config_command = PipelineFactory.get_yaml_config(
            yaml_config_path=yaml_config_path_run2,
            clean_truecased_files_names=clean_truecased_files_names_by_source_after_run1["Europarl"],
            language_codes=language_codes,
            folder_base_path=folder_base_path, # meme qu'avant, c'est sur ?
            run_number=2,
            train_steps= 10_000,
            valid_steps= 1_000,
            save_checkpoint_steps= 1_000,
        )

        pipeline.add_command(config_command)
        
        train_translate_evaluate_pipeline, model_paths_info = PipelineFactory.build_train_translate_evaluate_pipeline(
            yaml_config=config_command,
            clean_truecased_files_names=clean_truecased_files_names_by_source_after_run1["Europarl"],
            language_codes=language_codes,
            folder_base_path=folder_base_path,
            get_model_paths_info = True
        )

        # Evaluation:
        pipeline.add_command(
            # dont évaluation 1.1 => c'est à dire avec le domaine "classique"
            train_translate_evaluate_pipeline
        ).add_command(
            # évaluation 1.2 => c'est à dire hors domaine (on utilise EMEA)
            PipelineFactory.translate_and_evaluate(
                clean_truecased_files_names=clean_truecased_files_names_by_source_after_run1["EMEA"],
                model_paths_info=model_paths_info,
            )
        )

        return pipeline
        



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

