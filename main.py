from src.pipelines.pipeline_factory import PipelineFactory

if __name__ == "__main__":
    classic_pipeline = PipelineFactory.get_i2_all_run(
        folder_base_path= "./data/partieIII",
        language_codes= ["en", "fr"],
        yaml_config_path_run1= "./config/partieIII-europarl_en_fr.yaml",
        yaml_config_path_run2= "./config/partieIII-mix-europarl_emea_en_fr.yaml",
        useLemmatizer = True
    )
    classic_pipeline.execute()
