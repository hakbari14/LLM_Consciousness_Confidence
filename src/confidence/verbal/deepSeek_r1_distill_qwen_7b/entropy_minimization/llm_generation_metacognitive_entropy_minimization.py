from src.logger.llm_response.llm_response_inference_logger import llm_response_inference_logger
from src.confidence.llm_generation_metacognitive import llm_generation_metacognitive
from src.utils.enums_class import confidence_type_enum


class llm_generation_metacognitive_entropy_minimization(llm_generation_metacognitive): 

    def __init__(self, modelname):
        super().__init__(modelname)

    def create_llm_response_logger(self, run_number) -> llm_response_inference_logger:
        return llm_response_inference_logger(log_file_name = f'src/confidence/verbal/deepSeek_r1_distill_qwen_7b/entropy_minimization/run_{run_number}/llm_generation_metacognitive_entropy_minimization.csv')


t = llm_generation_metacognitive_entropy_minimization(modelname='hakbari/deepseek_r1_qwen_7B_iit_entropy_minimization_51')
t.run(from_run_number=1, to_run_number=2, confidence_type = confidence_type_enum.PROBABILITY)
t.run(from_run_number=2, to_run_number=3, confidence_type = confidence_type_enum.PROBABILITY)
t.run(from_run_number=3, to_run_number=4, confidence_type = confidence_type_enum.PROBABILITY)
t.run(from_run_number=4, to_run_number=5, confidence_type = confidence_type_enum.PROBABILITY)

# t.run(from_run_number=5, to_run_number=6, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=6, to_run_number=7, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=7, to_run_number=8, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=8, to_run_number=9, confidence_type = confidence_type_enum.LEVEL)

