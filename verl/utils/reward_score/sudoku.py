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
        
        # Normalize both answers by removing all whitespace
        model_normalized = ''.join(model_answer_cleaned.split())
        correct_normalized = ''.join(correct_solution.split())
        
        # Check if they match
        if model_normalized == correct_normalized:
            return 1
        
        # Also check if the model answer is valid (basic validation)
        # A valid 4x4 sudoku should have 16 numbers (4 rows x 4 columns)
        if len(model_normalized) == 16 and model_normalized.isdigit():
            # Verify each row, column, and 2x2 box contains 1-4
            grid = []
            for i in range(4):
                row = [int(model_normalized[i*4 + j]) for j in range(4)]
                grid.append(row)
            
            # Check if it's a valid sudoku solution
            if is_valid_sudoku(grid):
                # Even if valid, only return 1 if it matches the expected solution
                if model_normalized == correct_normalized:
                    return 1
        
        return 0
    except Exception as e:
        return 0

def is_valid_sudoku(grid):
    """
    Validate if a 4x4 sudoku grid is correctly solved.
    
    Args:
        grid: 2D list representing the sudoku grid
        
    Returns:
        True if valid, False otherwise
    """
    if len(grid) != 4:
        return False
    
    for row in grid:
        if len(row) != 4:
            return False
    
    # Check rows
    for row in grid:
        if sorted(row) != [1, 2, 3, 4]:
            return False
    
    # Check columns
    for col in range(4):
        column = [grid[row][col] for row in range(4)]
        if sorted(column) != [1, 2, 3, 4]:
            return False
    
    # Check 2x2 boxes
    for box_row in range(0, 4, 2):
        for box_col in range(0, 4, 2):
            box = []
            for i in range(2):
                for j in range(2):
                    box.append(grid[box_row + i][box_col + j])
            if sorted(box) != [1, 2, 3, 4]:
                return False
    
    return True

