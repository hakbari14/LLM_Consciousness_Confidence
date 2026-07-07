from src.logger.llm_response.llm_response_inference_logger import llm_response_inference_logger
from src.confidence.verbal.qwen3_8b.llm_generation_metacognitive_qwen3_8b import llm_generation_metacognitive_qwen3_8b


class llm_generation_metacognitive_no_training(llm_generation_metacognitive_qwen3_8b): 

    def __init__(self, modelname):
        super().__init__(modelname)

    def create_llm_response_logger(self, run_number) -> llm_response_inference_logger:
        return llm_response_inference_logger(log_file_name = f'src/confidence/verbal/qwen3_8b/no_training/run_{run_number}/llm_generation_metacognitive_no_training.csv')


t = llm_generation_metacognitive_no_training(modelname='Qwen/Qwen3-8B')
t.run(from_run_number=1, to_run_number=2)
# t.run(from_run_number=2, to_run_number=3)
# t.run(from_run_number=3, to_run_number=4)
# t.run(from_run_number=4, to_run_number=5)