from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset
from src.datasets.math.gsm8k_dataset import gsm8k_dataset
import re
import random
import pandas as pd 

class metacognitive_creation: 

    def __init__(self):
        self.dataset_columns = ['source', 'category','problem_id','question','correct_answer','incorrect_answer']
        
    
    def create_and_save(self) -> Dataset:
        features = Features({
            "question": Value("large_string"),
            "source": Value("string"),
            "category": Value("string"),
            "problem_id": Value("string"),
            "correct_answer": Value("string"),
            "incorrect_answer": Value("string"),
        })

        dataset = Dataset.from_dict(
            {
                "question": [],
                "source": [],
                "category": [],
                "problem_id": [],
                "correct_answer": [],
                "incorrect_answer": []
            },
            features=features
        )
        
        mmlu_dataset = self.extract_mmlu_dataset()
        dataset = concatenate_datasets([dataset, mmlu_dataset])
        mmlu_pro_dataset = self.extract_mmlu_pro_dataset()
        dataset = concatenate_datasets([dataset, mmlu_pro_dataset])
        truthfulqa_dataset = self.extract_truthfulqa_dataset()
        dataset = concatenate_datasets([dataset, truthfulqa_dataset])
        gpqa_dataset = self.extract_gpqa_dataset()
        dataset = concatenate_datasets([dataset, gpqa_dataset])

        gsm8k_dataset = self.extract_gsm8k_dataset()
        dataset = concatenate_datasets([dataset, gsm8k_dataset])
        math500_dataset = self.extract_math500_dataset()
        dataset = concatenate_datasets([dataset, math500_dataset])
        aime_dataset = self.extract_aime_dataset()
        dataset = concatenate_datasets([dataset, aime_dataset])
        countdown_dataset = self.extract_countdown_dataset()
        dataset = concatenate_datasets([dataset, countdown_dataset])
        dataset = dataset.add_column('unique_id', range(len(dataset)))
        dataset.to_csv("./data/metacognitive_dataset.csv") 

        df = dataset.to_pandas()        
        result = df.groupby("source").size().reset_index(name="count")       
        total_count = result["count"].sum()
        print(result)
        print("\nTotal count:", total_count)

        
    def extract_mmlu_dataset(self) -> Dataset:
        dataset_id = 'lighteval/mmlu'
        dataset: Dataset = load_dataset(dataset_id, "all")
        test_dataset: Dataset = dataset['test']
        
        seed = 42
        category_list = set(test_dataset["subject"])
        selected_indices = []
        dataset_size_per_category: int = 2
        for category in category_list:
            indices = [
                i for i, t in enumerate(test_dataset["subject"])
                if t == category
            ]

            if len(indices) <= dataset_size_per_category:
                selected_indices.extend(indices)
            else:
                rng = random.Random(seed)
                selected_indices.extend(rng.sample(indices, dataset_size_per_category))
                
        mmlu_dataset = test_dataset.select(selected_indices)
        mmlu_dataset = mmlu_dataset.map(self.format_mmlu)
        mmlu_dataset = self.filter_remove_columns_cast(mmlu_dataset)
        return mmlu_dataset

    def format_mmlu(self, x):
        subject = x["subject"]
        question = x["question"]
        choices = x["choices"]
        answer = x["answer"]
        correct_answer = choices[answer]

        choices_list: list[int] = [0,1,2,3]
        filtered = [i for i in choices_list if i != answer]        
        incorrect_answer = choices[random.choice(filtered)]

        return {
            "source": "mmlu",
            "category": subject,
            "problem_id": None,
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
        }

    def extract_mmlu_pro_dataset(self) -> Dataset:
        dataset_id = 'TIGER-Lab/MMLU-Pro'
        dataset: Dataset = load_dataset(dataset_id)
        test_dataset: Dataset = dataset['test']
        
        seed = 42
        dataset_size_per_category: int = 8
        category_list = set(test_dataset["category"])
        selected_indices = []
        for category in category_list:
            indices = [
                i for i, t in enumerate(test_dataset["category"])
                if t == category
            ]

            if len(indices) <= dataset_size_per_category:
                selected_indices.extend(indices)
            else:
                rng = random.Random(seed)
                selected_indices.extend(rng.sample(indices, dataset_size_per_category))

        mmlu_dataset_pro = test_dataset.select(selected_indices)        
        mmlu_dataset_pro = mmlu_dataset_pro.map(self.format_mmlu_pro)
        mmlu_dataset_pro = self.filter_remove_columns_cast(mmlu_dataset_pro)
        return mmlu_dataset_pro

    def format_mmlu_pro(self, x):
        question_id = x["question_id"]
        category = x["category"]
        question = x["question"]
        options = x["options"]
        answer_index = x["answer_index"]
        correct_answer = options[answer_index]

        options_list: list[int] = list(range(0, len(options)))
        filtered = [i for i in options_list if i != answer_index]        
        incorrect_answer = options[random.choice(filtered)]

        return {
            "source": "mmlu_pro",
            "category": category,
            "problem_id": str(question_id),
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
        }

    def extract_truthfulqa_dataset(self) -> Dataset:
        df = pd.read_parquet("data/EleutherAI_truthful_qa_mc.parquet")
        truthfulqa_dataset = Dataset.from_pandas(df.sample(frac=0.167, random_state=42))

        truthfulqa_dataset = truthfulqa_dataset.map(self.format_truthfulqa)
        truthfulqa_dataset = self.filter_remove_columns_cast(truthfulqa_dataset)
        return truthfulqa_dataset

    def format_truthfulqa(self, x):
        question: str = x['question']
        choices: list[str] = x['choices']
        label_index = x['label']

        correct_answer = choices[label_index]

        choices_list: list[int] = list(range(0, len(choices)))
        filtered = [i for i in choices_list if i != label_index]        
        incorrect_answer = choices[random.choice(filtered)]

        return {
            "source": "truthfulqa",
            "category": "truthfulqa",
            "problem_id": None,
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
        }

    def extract_gpqa_dataset(self) -> Dataset:
        df = pd.read_csv("data/gpqa_diamond.csv")
        gpqa_dataset = Dataset.from_pandas(df.sample(frac=0.58, random_state=42))

        gpqa_dataset = gpqa_dataset.map(self.format_gpqa)
        gpqa_dataset = self.filter_remove_columns_cast(gpqa_dataset)
        return gpqa_dataset

    def format_gpqa(self, x):
        question_id: str = str(x['id'])
        problem = x["problem"]
        category = x["subdomain"]
        answer = x["answer"]
        correct_answer = x["answer_content"]
        question = problem.split("Choices:")[0]

        choices: str = problem.split("Choices:")[1]
        incorrect_answer = self.get_random_other_option(choices, answer)

        return {
            "source": "gpqa",
            "category": category,
            "problem_id": question_id,
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
        }

    def extract_gsm8k_dataset(self) -> Dataset:
        dataset_id = "openai/gsm8k"
        dataset: Dataset = load_dataset(dataset_id, "main")
        gsm8k_dataset: Dataset = dataset["test"]
        gsm8k_dataset = gsm8k_dataset.train_test_split(test_size=0.086, seed=42, shuffle=True)['test']

        gsm8k_dataset = gsm8k_dataset.map(self.format_gsm8k)
        gsm8k_dataset = self.filter_remove_columns_cast(gsm8k_dataset)
        return gsm8k_dataset

    def format_gsm8k(self, x):
        question = x['question']
        correct_answer = str(gsm8k_dataset.gsm8k_answer_extraction(x["answer"]))
        incorrect_answer = self.generate_wrong_math_equation(correct_answer)

        return {
            "source": "gsm8k",
            "category": 'simple_math',
            "problem_id": None,
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
        }

    def extract_math500_dataset(self) -> Dataset:
        dataset_id = "HuggingFaceH4/MATH-500"
        dataset = load_dataset(dataset_id)
        math500_dataset = dataset['test']
        math500_dataset = math500_dataset.train_test_split(test_size=0.23, seed=42, shuffle=True)['test']

        math500_dataset = math500_dataset.map(self.format_math500)
        math500_dataset = self.filter_remove_columns_cast(math500_dataset)
        return math500_dataset

    def format_math500(self, x):
        unique_id = x['unique_id']
        category = x['subject']
        question = x['problem']
        correct_answer = str(x["answer"])
        incorrect_answer = self.generate_wrong_math_equation(correct_answer)

        return {
            "source": "math500",
            "category": category,
            "problem_id": str(unique_id),
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
        }

    def extract_aime_dataset(self) -> Dataset:
        dataset_id = "di-zhang-fdu/AIME_1983_2024" 
        dataset = load_dataset(dataset_id)
        aime_dataset = dataset['train']
        aime_dataset = aime_dataset.train_test_split(test_size=0.125, seed=42, shuffle=True)['test']

        aime_dataset = aime_dataset.map(self.format_aime)
        aime_dataset = self.filter_remove_columns_cast(aime_dataset)
        return aime_dataset

    def format_aime(self, x):
        problem_id = x["Problem Number"]
        question = x["Question"]
        try:
            correct_answer = str(x["Answer"])
            incorrect_answer = self.generate_wrong_math_equation(correct_answer)
        except:
            correct_answer = None
            incorrect_answer = None
            
        return {
            "source": "aime",
            "category": 'hard_math',
            "problem_id": str(problem_id),
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
        }

    def extract_countdown_dataset(self) -> Dataset:
        dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
        dataset = load_dataset(dataset_id)["train"].train_test_split(test_size=0.00024, seed=42)
        countdown_dataset = dataset["test"]

        countdown_dataset = countdown_dataset.map(self.format_countdown)
        countdown_dataset = self.filter_remove_columns_cast(countdown_dataset)
        return countdown_dataset

    def format_countdown(self, x):
        numbers = x['nums']
        question = f"Using the numbers {numbers}, create an equation that equals __target__. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once."        

        correct_answer = str(x["target"])
        incorrect_answer = self.generate_wrong_math_equation(correct_answer)

        return {
            "source": "countdown",
            "category": 'math',
            "problem_id": None,
            "question": question,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
        }

    def filter_remove_columns_cast(self, dataset) -> Dataset:
        dataset = dataset.filter(lambda x: x["correct_answer"] is not None and x["incorrect_answer"] is not None)        
        dataset = dataset.remove_columns([c for c in dataset.column_names if c not in self.dataset_columns])
        dataset = dataset.cast_column("question", Value("large_string"))        
        return dataset

    def generate_wrong_math_equation(self, latex_expr:str) -> str:
        origin_expr = str(latex_expr)
        numbers = re.findall(r'-?\d+\.?\d*', latex_expr)

        if not numbers:
            replacements = [
                (r'\+', '-'),
                (r'-', '+'),
                (r'\\times', r'\\div'),
                (r'\\div', r'\\times')
            ]

            for old, new in replacements:
                if re.search(old, latex_expr):
                    return re.sub(old, new, latex_expr, count=1)

            return latex_expr if latex_expr != origin_expr else None

        selected = random.choice(numbers)
        value = float(selected)
        delta = random.choice([-2, -1, 1, 2])
        new_value = value + delta

        if selected.isdigit():
            new_value = int(new_value)

        new_expr = latex_expr.replace(
            selected,
            str(new_value),
            1
        )
        return new_expr if new_expr != origin_expr else None
    
    def get_random_other_option(self, text, selected_option):
        pattern = r'([A-D])\.\s*(.+?)(?=\n[A-D]\.|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        options = {key: value.strip() for key, value in matches}

        remaining = {
            k: v for k, v in options.items()
            if k != selected_option.upper()
        }

        if not remaining:
            raise ValueError("No other options available")

        random_key = random.choice(list(remaining.keys()))
        return remaining[random_key]



d = metacognitive_creation()
d.create_and_save()
