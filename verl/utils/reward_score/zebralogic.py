import re
import random


def parse_answer(answer_str):
    """
    Parse and clean the answer string to extract the core answer.

    Handles formats like:
    - "A. Arnold" -> "A"
    - "A) Arnold" -> "A"
    - "A: Arnold" -> "A"
    - "Answer: A" -> "A"

    Args:
        answer_str: Raw answer string

    Returns:
        Cleaned answer string
    """
    if not answer_str:
        return None

    answer_str = str(answer_str).strip()

    # Pattern 1: Single letter followed by punctuation and text
    match = re.match(r'^([A-Za-z])[.):]\s*\w+', answer_str)
    if match:
        return match.group(1).upper()

    # Pattern 2: "Answer: A" format
    match = re.search(r'answer[:\s]+([A-Za-z])\b',
                      answer_str, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: Single letter (convert to uppercase)
    if len(answer_str) == 1 and answer_str.isalpha():
        return answer_str.upper()

    # Pattern 4: Return the original if already clean
    return answer_str


def extract_solution(solution_str):
    # Use DOTALL flag to make . match newlines as well
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
        # Parse the answer to handle various formats
        final_answer = parse_answer(final_answer)
    else:
        final_answer = None

    return final_answer


def compare_answer(model_answer, correct_answer, choices=None):
    """
    Compare the model's ZebraLogic answer with the correct answer.

    Args:
        model_answer: String containing the model's proposed answer
            (could be option letter like 'A')
        correct_answer: String containing the correct answer
        choices: Optional list/array of choices to map option letters
            to actual answers

    Returns:
        1 if match (case-insensitive, ignoring whitespace), 0 otherwise
    """
    if not model_answer:
        return 0
    
    try:
        # First parse the model answer to handle various formats
        model_parsed = parse_answer(model_answer)
        if not model_parsed:
            return 0

        # Clean and normalize both answers
        model_cleaned = str(model_parsed).strip().lower()
        correct_cleaned = str(correct_answer).strip().lower()

        # Remove common prefixes/suffixes from correct answer
        prefixes = ['answer:', 'solution:', 'result:', 'the answer is',
                    'the correct answer is']
        for prefix in prefixes:
            if model_cleaned.startswith(prefix):
                model_cleaned = model_cleaned[len(prefix):].strip()

        # Remove quotes
        model_cleaned = model_cleaned.strip('"\'')
        correct_cleaned = correct_cleaned.strip('"\'')

        # If choices provided and model answer is single letter,
        # map it to actual choice
        if (choices is not None and len(model_cleaned) == 1 and
                model_cleaned.isalpha()):
            option_index = ord(model_cleaned) - ord('a')
            if 0 <= option_index < len(choices):
                # Map option letter to actual answer
                model_cleaned = str(choices[option_index]).strip().lower()
        
        # Direct comparison
        if model_cleaned == correct_cleaned:
            return 1
        
        # Check if model answer contains the correct answer
        if correct_cleaned in model_cleaned:
            return 1
        
        # Check if correct answer contains model answer (for partial matches)
        if model_cleaned in correct_cleaned:
            return 1
        
        return 0
    except Exception:
        return 0


def compute_score(solution_str, ground_truth, method='strict',
                  format_score=0.1, score=1., choices=None):
    """The scoring function for ZebraLogic.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced
    fine-tuning." ACL 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth dictionary with 'label' key
        method: deprecated, unused
        format_score: DEPRECATED - no longer used, only correct
            answers get score
        score: the score for the correct answer (default 1.0)
        choices: Optional list/array of answer choices for mapping
            option letters to actual answers
    """

    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    if do_print:
        print("--------------------------------")
        print("Ground truth: {} | Extracted answer: {}".format(
            ground_truth, answer))
        print("Choices: {}".format(choices))
        print("Solution string: {}...".format(solution_str[:200]))

    if answer is None:
        if do_print:
            print("No answer found")
        return 0
    else:
        # Use the more robust compare_answer function
        correct_answer = ground_truth["label"]
        is_correct = compare_answer(answer, correct_answer,
                                     choices=choices)

        if is_correct:
            if do_print:
                print("Correct answer: {}".format(answer))
            return score
        else:
            if do_print:
                print("Incorrect answer {} | Ground truth: {}".format(
                    answer, correct_answer))
            # No partial credit - only correct answers get points
            return 0
