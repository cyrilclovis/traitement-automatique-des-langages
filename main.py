from src.pipelines.pipeline_factory import PipelineFactory

if __name__ == "__main__":
    classic_pipeline = PipelineFactory.get_classic_pipeline()
    classic_pipeline.execute()
