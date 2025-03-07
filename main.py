from src.pipelines.pipeline_factory import PipelineFactory

if __name__ == "__main__":
    classic_pipeline = PipelineFactory.get_i2_all_run(
        folder_base_path= "./data/partieII",
        language_codes= ["en", "fr"],
        yaml_config_path_run1= "./config/partieII-europarl_en_fr.yaml",
        yaml_config_path_run2= "./config/partieII-mix-europarl_emea_en_fr.yaml"
    )
    classic_pipeline.execute()
