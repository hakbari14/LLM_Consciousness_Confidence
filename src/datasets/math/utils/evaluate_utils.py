import sympy
from sympy.parsing.latex import parse_latex
from math_verify import parse, verify
import re

def remove_punctuations_and_latex(text):
    # Define a regular expression pattern to match punctuations and LaTeX symbols
    pattern = r'[^\w\s]'
    # Use re.sub() to replace matched characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    cleaned_text = cleaned_text.replace(" ", "")
    return cleaned_text

def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    if x1 == x2:
        return True
    try:
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                # eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                # eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                pass
                # eval_logger.debug(
                #     f"Had some trouble simplifying when comparing {x1} and {x2}"
                # )
    except Exception as e:
        # eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False

def use_math_verify(gold, answer):
    gold = str(gold)
    answer = str(answer)
    if gold=="" or answer=="":
        return False
        
    gold_orig = gold
    answer_orig = answer
    
    # Text responses may be in the form abc or \text{xyz}
    gold = re.sub(r'\\text\{([^}]*)\}', r'\1', gold)
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)

    # Brackets may be in the form "(" or "\left(" AND ")" or "\right)"
    gold = re.sub(r'\\left\(', r'(', gold)
    gold = re.sub(r'\\right\)', r')', gold)
    answer = re.sub(r'\\left\(', r'(', answer)
    answer = re.sub(r'\\right\)', r')', answer)

    # Replace "_" or "," between 2 digits with "". This ensures equality between 10_000 and 10,000 and 10000
    gold = re.sub(r'(?<=\d)[_,](?=\d)', '', gold)
    answer = re.sub(r'(?<=\d)[_,](?=\d)', '', answer)
    
    if gold==answer:
        return True
    if gold.replace(" ", "")==answer.replace(" ", "") and gold.replace(" ", "")!="":
        return True
    if verify(parse(gold), parse(answer)):
        return True
    if is_equiv(gold, answer):
        return True
        
    if is_equiv(gold_orig, answer_orig):
        return True
    if verify(parse(gold_orig), parse(answer_orig)):
        return True
    gold_clean = remove_punctuations_and_latex(gold)
    answer_clean = remove_punctuations_and_latex(answer)
    if gold_clean!="" and gold_clean==answer_clean:
        return True
    return False

def extract_boxed_answer(text):
    if text is None: 
        return None
    
    # extract answer wrapped in \boxed{} from models' output
    matches = re.finditer(r'\\boxed{', text)
    if not matches:
        return None

    last_match = None
    for m in matches: 
        last_match = m

    if last_match is None: 
        return None
    
    start_index = last_match.end()
    end_index = start_index
    stack = 1
    while stack > 0 and end_index < len(text):
        if text[end_index] == '{':
            stack += 1
        elif text[end_index] == '}':
            stack -= 1
        end_index += 1
    if stack == 0:
        content = text[start_index:end_index - 1]
        if not content:
            return text
        else:
            content = normalize_answer(content)
            return content
    return text
    

def normalize_answer(final_answer):
    special_signal_map = {
        "\\left": "",
        "\\right": "",
        "∶": ":",
        "，": ",",
        "$": "",
        "\\approx": "=",
        "\\simeq": "=",
        "\\sim": "=",
        "^\\prime": "'",
        "^{\\prime}": "'",
        "^\\circ": "",
        "\\dfrac": "\\frac",
        "%": "",
    }
    for signal in special_signal_map:
        final_answer = final_answer.replace(signal, special_signal_map[signal])
    final_answer = re.sub(r'\\(?:mathrm|mathbf)\{~?([^}]*)\}', '\\1', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(
        r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
        r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.strip()
    final_answer = final_answer.strip("$")
    final_answer = final_answer.strip()
    return final_answer

