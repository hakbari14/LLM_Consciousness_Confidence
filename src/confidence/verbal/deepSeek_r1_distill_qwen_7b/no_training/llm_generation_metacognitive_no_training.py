from src.logger.llm_response.llm_response_inference_logger import llm_response_inference_logger
from src.confidence.llm_generation_metacognitive import llm_generation_metacognitive
from src.utils.enums_class import confidence_type_enum


class llm_generation_metacognitive_no_training(llm_generation_metacognitive): 

    def __init__(self, modelname):
        super().__init__(modelname)

    def create_llm_response_logger(self, run_number) -> llm_response_inference_logger:
        return llm_response_inference_logger(log_file_name = f'src/confidence/verbal/deepSeek_r1_distill_qwen_7b/no_training/run_{run_number}/llm_generation_metacognitive_no_training.csv')


t = llm_generation_metacognitive_no_training(modelname='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
# t.run(from_run_number=1, to_run_number=2, confidence_type = confidence_type_enum.PROBABILITY)
# t.run(from_run_number=2, to_run_number=3, confidence_type = confidence_type_enum.PROBABILITY)
# t.run(from_run_number=3, to_run_number=4, confidence_type = confidence_type_enum.PROBABILITY)
# t.run(from_run_number=4, to_run_number=5, confidence_type = confidence_type_enum.PROBABILITY)

# t.run(from_run_number=5, to_run_number=6, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=6, to_run_number=7, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=7, to_run_number=8, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=8, to_run_number=9, confidence_type = confidence_type_enum.LEVEL)

# t.run(from_run_number=9, to_run_number=10, confidence_type = confidence_type_enum.PROBABILITY)
# t.run(from_run_number=10, to_run_number=11, confidence_type = confidence_type_enum.PROBABILITY)
# t.run(from_run_number=11, to_run_number=12, confidence_type = confidence_type_enum.PROBABILITY)
t.run(from_run_number=12, to_run_number=13, confidence_type = confidence_type_enum.PROBABILITY)

# t.run(from_run_number=13, to_run_number=14, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=14, to_run_number=15, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=15, to_run_number=16, confidence_type = confidence_type_enum.LEVEL)
# t.run(from_run_number=16, to_run_number=17, confidence_type = confidence_type_enum.LEVEL)

