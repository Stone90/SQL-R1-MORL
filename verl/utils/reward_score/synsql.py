import ast
import re
import os
import sqlite3
from typing import Dict, Tuple, Optional
from func_timeout import func_timeout
from .exec_eval import eval_exec_match, get_cursor_from_path, postprocess

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


def _explain_plan_cost(cursor, sql: str) -> Optional[float]:
    """Run EXPLAIN QUERY PLAN and return a cost score based on operation types.

    Lower cost = more efficient plan. Returns None if the query fails to explain.

    Scoring:
        - SCAN TABLE (full table scan): 10 points each
        - USE TEMP B-TREE (temp sorting/grouping): 5 points each
        - SEARCH TABLE USING INDEX (index lookup): 1 point each
        - Other operations: 0 points
    """
    try:
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        rows = cursor.fetchall()
    except Exception:
        return None

    cost = 0.0
    for row in rows:
        detail = str(row[-1]).upper() if row else ""
        if "SCAN TABLE" in detail:
            cost += 10.0
        elif "SEARCH TABLE" in detail or "SEARCH SUBQUERY" in detail:
            cost += 1.0
        if "USE TEMP B-TREE" in detail:
            cost += 5.0
    return cost


def _compute_efficiency_reward(db_path: str, pred_sql: str, gold_sql: str, exec_status: str) -> float:
    """Compute efficiency reward by comparing EXPLAIN QUERY PLAN costs.

    Returns a reward in [0, 1]:
        - 1.0 if pred plan cost <= gold plan cost (at least as efficient)
        - Proportionally penalized if pred is less efficient than gold
        - 0.0 if the query cannot be explained or is unexecutable
    Only awarded when exec_status is Match or Mismatch (SQL must at least run).
    """
    if exec_status not in ('Match', 'Mismatch'):
        return 0.0

    try:
        cursor = get_cursor_from_path(db_path)
        pred_cost = _explain_plan_cost(cursor, postprocess(pred_sql))
        gold_cost = _explain_plan_cost(cursor, postprocess(gold_sql))
        cursor.close()
        cursor.connection.close()
    except Exception:
        return 0.0

    if pred_cost is None or gold_cost is None:
        return 0.0

    # Pred is at least as efficient as gold
    if pred_cost <= gold_cost:
        reward = 1.0
    else:
        # Penalize proportionally: reward = gold_cost / pred_cost
        reward = gold_cost / pred_cost if pred_cost > 0 else 0.0

    # Half credit for Mismatch (executable but wrong result)
    if exec_status == 'Mismatch':
        reward *= 0.5

    return reward


def compute_score(solution_str: str, ground_truth: Dict):
    """
    Computes the accuracy and efficiency rewards for a given SQL solution.

    Returns:
        A tuple of (accuracy_reward, efficiency_reward).
    """
    FORMAT_REWARD = 1
    EXEC_REWARD = 2
    RESULT_REWARD = 3

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

    if not db_id or not gold_sql:
        return -1.0, 0.0

    answer_text, think_text, processed_str = extract_solution(solution_str)
    pred_sql, format_correct = validate_response_structure(answer_text, processed_str)

    format_score = FORMAT_REWARD if format_correct else -0.5
    exec_score = 0
    result_score = 0
    exec_status = "Not Attempted"

    db_base = os.environ.get('SYNSQL_DB_DIR', '/workspace/synsql_data/data/SynSQL-2.5M/databases')
    db_path = os.path.join(db_base, db_id, f"{db_id}.sqlite")

    if pred_sql:
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

    # Efficiency: compare EXPLAIN QUERY PLAN costs of pred vs gold
    efficiency_reward = 0.0
    if pred_sql and os.path.exists(db_path):
        efficiency_reward = _compute_efficiency_reward(db_path, pred_sql, gold_sql, exec_status)

    accuracy_reward = format_score + exec_score + result_score

    return accuracy_reward, efficiency_reward
