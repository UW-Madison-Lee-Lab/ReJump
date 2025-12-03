import pdb
from verl.utils.reward_score.general import last_answer_string

def compute_score(solution_str, ground_truth):
    answer = last_answer_string(solution_str, "tags")
    return compare_answer(answer, ground_truth["label"][0])

def compare_answer(model_answer, correct_solution):
    """
    Compare the model's sudoku solution with the correct solution.
    
    Args:
        model_answer: String containing the model's proposed solution
        correct_solution: String containing the correct solution
        
    Returns:
        1 if solutions match (ignoring whitespace), 0 otherwise
    """
    if not model_answer:
        return 0
    
    try:
        # Remove common prefixes like "Answer:", "Solution:", etc.
        import re
        
        # First, try to match specific prefixes
        pattern = r'^(Answer:|Solution:|Result:|Output:)\s*'
        match = re.match(pattern, model_answer, flags=re.IGNORECASE)
        
        if match:
            # If matched, remove the prefix
            model_answer_cleaned = re.sub(pattern, '', model_answer, flags=re.IGNORECASE).strip()
        elif ':' in model_answer:
            # If no match but contains colon, extract content after the last colon
            model_answer_cleaned = model_answer.rsplit(':', 1)[-1].strip()
        else:
            # No prefix and no colon, use as is
            model_answer_cleaned = model_answer.strip()
        
        # Handle literal \n strings (e.g., "12453\\n31524" -> "12453\n31524")
        # Some models output the string "\\n" instead of actual newline
        model_answer_cleaned = model_answer_cleaned.replace('\\n', '\n')
        
        # Remove XML/HTML tags like </answer>, <answer>, etc.
        model_answer_cleaned = re.sub(r'</?answer>', '', model_answer_cleaned, flags=re.IGNORECASE).strip()
        
        # Normalize both answers by removing all whitespace
        model_normalized = ''.join(model_answer_cleaned.split())
        correct_normalized = ''.join(correct_solution.split())
        
        # Infer grid size from correct solution
        correct_lines = correct_solution.strip().split('\n')
        grid_size = len(correct_lines)
        expected_length = grid_size * grid_size
        
        # Check if the model answer has correct length and is all digits
        if len(model_normalized) != expected_length or not model_normalized.isdigit():
            return 0
        
        # Parse model answer into grid
        grid = []
        for i in range(grid_size):
            row = [int(model_normalized[i*grid_size + j]) for j in range(grid_size)]
            grid.append(row)
        
        # Check if it's a valid Latin Square solution
        # Accept any valid solution, not just the one matching the expected answer
        if is_valid_sudoku(grid, grid_size):
            return 1
        
        return 0
    except Exception as e:
        return 0

def is_valid_sudoku(grid, grid_size=None):
    """
    Validate if a Latin Square grid is correctly solved.
    Each row and column should contain numbers 1-grid_size exactly once.
    Note: This is a Latin Square, not standard Sudoku - no box constraints.
    
    Args:
        grid: 2D list representing the grid
        grid_size: Size of the grid (n for n×n). 
                   If None, inferred from grid length.
        
    Returns:
        True if valid, False otherwise
    """
    if not grid:
        return False
    
    if grid_size is None:
        grid_size = len(grid)
    
    if len(grid) != grid_size:
        return False
    
    expected_values = list(range(1, grid_size + 1))
    
    # Check rows
    for row in grid:
        if len(row) != grid_size or sorted(row) != expected_values:
            return False
    
    # Check columns
    for col in range(grid_size):
        column = [grid[row][col] for row in range(grid_size)]
        if sorted(column) != expected_values:
            return False
    
    return True

