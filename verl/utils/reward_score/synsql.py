import ast
import re
import os
from typing import Dict, Tuple, Optional
from func_timeout import func_timeout
from .exec_eval import eval_exec_match

def extract_solution(solution_str: str) -> Tuple[Optional[str], Optional[str], str]:
    """Extracts the final answer from the model's response string safely."""
    processed_str = solution_str

    # Attempt to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "assistant\n" in solution_str.lower():
        processed_str = re.split(r'assistant\n', solution_str, flags=re.IGNORECASE)[-1]

    # Regex patterns
    answer_pattern = r'<answer>(.*?)</answer>'
    think_pattern = r'<think>(.*?)</think>'

    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    think_matches = list(re.finditer(think_pattern, processed_str, re.DOTALL))

    final_think = think_matches[-1].group(1).strip() if think_matches else ""
    final_answer = matches[-1].group(1).strip() if matches else None

    return final_answer, final_think, processed_str

def parse_sql_from_answer(answer_text: str) -> Optional[str]:
    if not answer_text:
        return None
    sql_pattern = r'```sql(.*?)```'
    matches = list(re.finditer(sql_pattern, answer_text, re.DOTALL))
    if not matches:
        # Fallback: check for any code block if sql tag is missing
        code_pattern = r'```(.*?)```'
        matches = list(re.finditer(code_pattern, answer_text, re.DOTALL))

    return matches[-1].group(1).strip() if matches else answer_text.strip()

def validate_response_structure(answer_str: str, processed_str: str) -> Tuple[Optional[str], bool]:
    if not answer_str:
        return None, False

    tags = ['<think>', '</think>', '<answer>', '</answer>']
    positions = {tag: processed_str.find(tag) for tag in tags}

    # Check if all tags exist and are in order
    if all(pos != -1 for pos in positions.values()) and \
       positions['<think>'] < positions['</think>'] < positions['<answer>'] < positions['</answer>']:
        pred_sql = parse_sql_from_answer(answer_str)
        return pred_sql, True

    return None, False


def compute_score(solution_str: str, ground_truth: Dict):
    """
    Computes the accuracy and efficiency rewards for a given SQL solution.

    Returns:
        A tuple of (accuracy_reward, efficiency_reward).
    """
    FORMAT_REWARD = 1
    EXEC_REWARD = 2
    RESULT_REWARD = 3
    LIMIT_LENGTH = 2048

    if isinstance(ground_truth, str):
        try:
            ground_truth = ast.literal_eval(ground_truth)
        except (ValueError, SyntaxError):
            return -1.0, 0.0

    inner_data = ground_truth.get('ground_truth', {})

    if isinstance(inner_data, dict):
        db_id = inner_data.get('db_id')
        gold_sql = inner_data.get('sql')
    else:
        db_id = ground_truth.get('db_id')
        gold_sql = ground_truth.get('sql')

    if not db_id:
        return -1.0, 0.0

    answer_text, think_text, processed_str = extract_solution(solution_str)
    pred_sql, format_correct = validate_response_structure(answer_text, processed_str)

    format_score = FORMAT_REWARD if format_correct else -0.5
    exec_score = 0
    result_score = 0
    exec_status = "Not Attempted"

    if pred_sql:
        db_base = os.environ.get('SYNSQL_DB_DIR', '/workspace/synsql_data/data/SynSQL-2.5M/databases')
        db_path = os.path.join(db_base, db_id, f"{db_id}.sqlite")

        try:
            if not os.path.exists(db_path):
                exec_status = 'Unexecutable'
            else:
                exec_status = func_timeout(10, eval_exec_match, args=(db_path, pred_sql, gold_sql, 0, False, False))
        except Exception as e:
            exec_status = 'Unexecutable'

        if exec_status == 'Match':
            exec_score, result_score = EXEC_REWARD, RESULT_REWARD
        elif exec_status == 'Mismatch':
            exec_score, result_score = EXEC_REWARD, -1
        else:
            exec_score = -1

    efficiency_reward = 0.0
    if format_correct and pred_sql:
        actual_think = think_text if think_text else ""
        actual_answer = answer_text if answer_text else ""
        pos_length = len(actual_think) + len(actual_answer)
        brevity = 1.0 - min(pos_length / LIMIT_LENGTH, 1.0)

        if exec_status == 'Match':
            # Full brevity reward for correct SQL
            efficiency_reward = brevity
        elif exec_status == 'Mismatch':
            # Partial credit: executable but wrong -- still reward brevity at half weight
            efficiency_reward = 0.5 * brevity
        # Unexecutable: no efficiency reward (keep 0)

    accuracy_reward = format_score + exec_score + result_score

    return accuracy_reward, efficiency_reward
