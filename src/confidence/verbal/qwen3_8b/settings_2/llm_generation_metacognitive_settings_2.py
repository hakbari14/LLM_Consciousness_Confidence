from src.logger.llm_response.llm_response_inference_logger import llm_response_inference_logger
from src.confidence.verbal.qwen3_8b.llm_generation_metacognitive_qwen3_8b import llm_generation_metacognitive_qwen3_8b


class llm_generation_metacognitive_settings_2(llm_generation_metacognitive_qwen3_8b): 

    def __init__(self, modelname):
        super().__init__(modelname)

    def create_llm_response_logger(self, run_number) -> llm_response_inference_logger:
        return llm_response_inference_logger(log_file_name = f'src/confidence/verbal/qwen3_8b/settings_2/run_{run_number}/llm_generation_metacognitive_settings_2.csv')


t = llm_generation_metacognitive_settings_2(modelname='./live_logs/settings_2/checkpoint-600-HF')
t.run(from_run_number=5, to_run_number=6)
# t.run(from_run_number=6, to_run_number=7)
# t.run(from_run_number=7, to_run_number=8)
# t.run(from_run_number=8, to_run_number=9)