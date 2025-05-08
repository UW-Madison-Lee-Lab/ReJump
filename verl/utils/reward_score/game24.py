import pdb
from verl.utils.reward_score.general import last_answer_string

def compute_score(solution_str, ground_truth):
    answer = last_answer_string(solution_str, "tags")
    if answer and "=" in answer:
        try:
            if eval(answer.split("=")[0].strip()) == 24:
                return 1
        except Exception as e:
            return 0

    return 0