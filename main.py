from src.pipelines.pipeline_factory import PipelineFactory

if __name__ == "__main__":
    classic_pipeline = PipelineFactory.get_pipeline_i1()
    classic_pipeline.execute()
