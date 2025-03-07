import argparse
from src.pipelines.pipeline_factory import PipelineFactory

def run_partie_i():
    """Mise en jambe sur le corpus fourni par le projet."""
    PipelineFactory.get_pipeline_i1().execute()

def run_partie_ii():
    """Évaluation sur des corpus parallèles en formes fléchies à large échelle."""
    PipelineFactory.get_i2_all_run(
        folder_base_path= "./data/partieII",
        language_codes= ["en", "fr"],
        yaml_config_path_run1= "./config/partieII-europarl_en_fr.yaml",
        yaml_config_path_run2= "./config/partieII-mix-europarl_emea_en_fr.yaml"
    ).execute()

def run_partie_iii():
    """Évaluation sur des corpus parallèles en lemmes à large échelle."""
    PipelineFactory.get_i2_all_run(
        folder_base_path="./data/partieIII",
        language_codes=["en", "fr"],
        yaml_config_path_run1="./config/partieIII-europarl_en_fr.yaml",
        yaml_config_path_run2="./config/partieIII-mix-europarl_emea_en_fr.yaml",
        useLemmatizer=True
    ).execute()

def main():
    parser = argparse.ArgumentParser(description="Pipeline à exécuter")
    parser.add_argument("part", choices=["I", "II", "III"], help="Sélectionner la partie du projet à exécuter :\n"
                        "I - Mise en jambe sur le corpus fourni par le projet\n"
                        "II - Évaluation sur des corpus parallèles en formes fléchies à large échelle\n"
                        "III - Évaluation sur des corpus parallèles en lemmes à large échelle")
    args = parser.parse_args()
    
    if args.part == "I":
        run_partie_i()
    elif args.part == "II":
        run_partie_ii()
    elif args.part == "III":
        run_partie_iii()
    
if __name__ == "__main__":
    main()
