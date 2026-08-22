from src.datasets.dataset_handler import dataset_handler
from src.datasets.dataset_config import dataset_config
from src.utils.enums_class import llm_pipeline_type_enum
from datasets import Dataset
from datasets import load_dataset
import re



class countdown_dataset(dataset_handler): 

    def __init__(self, config):
        super().__init__(config)
        self.dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
        self.dataset = load_dataset(self.dataset_id)["train"].train_test_split(test_size=0.001, seed=42)
        self.train_dataset = Dataset.from_dict({"prompt": [], "target": [], "problem_id" : []})
        self.test_dataset = self.dataset["test"]

        if config.get_ratio_test_dataset_size() is not None: 
            self.test_dataset = self.test_dataset.train_test_split(test_size=config.get_ratio_test_dataset_size(), seed=42, shuffle=True)['test']

        self.instruction = self.prompt_config.get('Math', 'countdown') 

    def final_answer_extraction(self, prompt, solution, target):
        patterns = [
            r'(?i)<answer>(.*?)</answer>',
            r'(?i)\\?oxed\s*\{(.*?)\}',            
        ]

        for pattern in patterns:
            pattern_regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
            matches = pattern_regex.findall(solution)
            if not matches or len(matches) == 0: continue
            
            equation = matches[-1].strip()
            equation = equation.replace('\\times', '*')
            equation = equation.replace('\\div', '/')
            if "=" in equation:
                equation = equation.split("=", 1)[0]
            
            return equation
    
        return None

    def generate_model_prompt(self, x):
        numbers = x['nums']
        target = x['target']
        problem_id = None
        question = f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."        
        
        r1_prefix = [{ 
            "role": "user",
            "content": question
        },]

        return {
                "prompt": self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), 
                "target": target, 
                "nums": numbers,
                "question": question,
                "problem_id": problem_id
                }

    def generate_model_prompt_chain_of_thought(self, x: dict, partial_cot: str) -> str:
        numbers = x['nums']
        target = x['target']
        question = f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."        

        prefix = [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": partial_cot
            },
        ]

        return self.tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)

    def generate_wrong_answer(self, latex_expr:str) -> str:
        return None

    def final_answer_confidence_extraction(self, prompt, completion, target):
        return None

    def generate_model_prompt_confidence(self, x):
        return None

    def generate_another_prompt_confidence(self, question: str, answer: str) -> str:
        return None

    def extract_another_confidence(self, solution: str) -> float:
        return None

    def verify_final_answer(self, target, equation):
        equation_expr = self.normalize_math_text(equation)

        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_expr):
           return False, None
       
        result = eval(equation_expr, {"__builtins__": None}, {})
        return abs(float(result) - float(target)) < 1e-5, result

    def normalize_math_text(self, text: str) -> str:
        replacements = {
            # Multiplication
            '×': '*',
            '✕': '*',
            '✖': '*',
            '⨯': '*',
            '⨉': '*',
            '⋅': '*',
            '∙': '*',
            '·': '*',
            '＊': '*',

            # Division
            '÷': '/',
            '∕': '/',
            '⁄': '/',
            '／': '/',

            # Plus
            '＋': '+',

            # Minus / hyphen
            '−': '-',
            '–': '-',
            '—': '-',
            '﹣': '-',
            '－': '-',

            # Equal
            '＝': '=',

            # Comparison
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '＜': '<',
            '＞': '>',

            # Parentheses
            '（': '(',
            '）': ')',
            '［': '[',
            '］': ']',
            '｛': '{',
            '｝': '}',

            # Comma / decimal separators
            '，': ',',
            '．': '.',

            # Colon
            '：': ':',

            # Percent
            '％': '%',

            # Power-related
            '∧': '^',

            # Square root
            '√': 'sqrt',

            # Infinity
            '∞': 'inf',

            # Common spaces
            '\u00A0': ' ',   # non-breaking space
            '\u2009': ' ',   # thin space
            '\u200A': ' ',   # hair space
            '\u202F': ' ',   # narrow no-break space
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Convert Unicode superscript digits to normal digits
        superscripts = str.maketrans(
            '⁰¹²³⁴⁵⁶⁷⁸⁹',
            '0123456789'
        )
        text = text.translate(superscripts)

        # Convert Unicode subscript digits
        subscripts = str.maketrans(
            '₀₁₂₃₄₅₆₇₈₉',
            '0123456789'
        )
        text = text.translate(subscripts)

        # Normalize multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)

        # Remove spaces around operators
        text = re.sub(r'\s*([+\-*/=<>])\s*', r'\1', text)

        return text.strip()


# config = dataset_config('Qwen/Qwen2.5-1.5B')
# config.set_pipeline_type(llm_pipeline_type_enum.INFERENCE)
# d = countdown_dataset(config)
# train_dataset, test_dataset = d.preprocess_dataset()
# print(len(train_dataset))
# print(len(test_dataset))
