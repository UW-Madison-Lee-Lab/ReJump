import pandas as pd

# Set pandas display options to show full content
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

def read_parquet(file_path):
    """
    Read a parquet file and return a pandas DataFrame
    
    Args:
        file_path (str): Path to the parquet file
        
    Returns:
        pd.DataFrame: DataFrame containing the parquet data
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully read parquet file: {file_path}")
        print(f"DataFrame shape: {df.shape}")
        #print("\nFirst few rows of the data:")
        #print(df.head())
        return df
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return None

def get_first_value(file_path):
    """
    Read the first value from a parquet file
    
    Args:
        file_path (str): Path to the parquet file
        
    Returns:
        The first value from the DataFrame
    """
    try:
        df = pd.read_parquet(file_path)
        first_value = df.iloc[0, 0]  # Get first value from first row and first column
        print(f"First value in the parquet file: {first_value}")
        return first_value
    except Exception as e:
        print(f"Error reading first value: {e}")
        return None

def print_columns(file_path, columns):
    """
    Print specific columns from a parquet file
    
    Args:
        file_path (str): Path to the parquet file
        columns (list): List of column names to print
    """
    try:
        df = pd.read_parquet(file_path)
        
        # Check if all requested columns exist
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: The following columns do not exist: {missing_columns}")
            return
        
        # Print selected columns
        print(f"\nSelected columns: {columns}")
        print("\nFirst few rows of selected columns:")
        print(df[columns].head())
        
        # Print basic statistics for numeric columns
        numeric_columns = df[columns].select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_columns) > 0:
            print("\nBasic statistics for numeric columns:")
            print(df[numeric_columns].describe())
            
    except Exception as e:
        print(f"Error printing columns: {e}")

def show_columns(file_path):
    """
    Show all column names from a parquet file
    
    Args:
        file_path (str): Path to the parquet file
    """
    try:
        df = pd.read_parquet(file_path)
        print("\nAvailable columns in the parquet file:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
        print(f"\nTotal number of columns: {len(df.columns)}")
    except Exception as e:
        print(f"Error showing columns: {e}")

def print_full_column(file_path, column_name):
    """
    Print all values of a specific column
    
    Args:
        file_path (str): Path to the parquet file
        column_name (str): Name of the column to print
    """
    try:
        df = pd.read_parquet(file_path)
        
        if column_name not in df.columns:
            print(f"Warning: Column '{column_name}' does not exist")
            return
            
        print(f"\nAll values in column '{column_name}':")
        print("-" * 50)
        for idx, value in enumerate(df[column_name]):
            print(f"Row {idx}:")
            print(value)
            print("-" * 30)
        print(f"Total number of rows: {len(df)}")
        
    except Exception as e:
        print(f"Error printing full column: {e}")

def print_specific_rows(file_path, rows, columns=None):
    """
    Print specific rows from a parquet file
    
    Args:
        file_path (str): Path to the parquet file
        rows (list): List of row indices to print
        columns (list, optional): List of column names to print. If None, prints all columns.
    """
    try:
        df = pd.read_parquet(file_path)
        
        # Validate row indices
        max_row = len(df) - 1
        invalid_rows = [row for row in rows if row < 0 or row > max_row]
        if invalid_rows:
            print(f"Warning: Invalid row indices: {invalid_rows}")
            return
            
        # Validate columns if specified
        if columns:
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: The following columns do not exist: {missing_columns}")
                return
            df = df[columns]
            
        print(f"\nSelected rows: {rows}")
        print("-" * 50)
        for row in rows:
            print(f"\nRow {row}:")
            for col in df.columns:
                print(f"{col}:")
                temp = df.iloc[row][col]
                if isinstance(temp, dict):
                    print(temp)
                else:
                    print(temp.item(0))
                print("-" * 20)
            print("-" * 30)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error printing specific rows: {e}")

if __name__ == "__main__":
    # Example usage
    file_path = "/pvc/home-mjlee/workspace/wjkang/neurips/liftr/results/deepseek-ai-deepseek-reasoner/circles_50_shot_base_reslen_3046_nsamples_500_noise_0.0_flip_rate_0.0/global_step_0/test.parquet"  # Replace with your parquet file path
    
    # Show all column names first
    show_columns(file_path)
    
    # Read entire DataFrame
    df = read_parquet(file_path)
    
    # Get first value
    #first_value = get_first_value(file_path)
    
    # Print specific columns
    #columns_to_print = ["ground_truths", "responses", "reasonings"]  # Replace with your column names
    #columns_to_print = ["reasonings"]
    columns_to_print = ["ground_truths", "responses"]
    #print_columns(file_path, columns_to_print)
    
    # Print full column
    #print_full_column(file_path, "ground_truths")  # Replace with your column name
    
    # Print specific rows
    rows_to_print = [16,17,18,19,20]  # Replace with your desired row indices
    #print_specific_rows(file_path, rows_to_print)
    
    # Print specific rows with specific columns
    print_specific_rows(file_path, rows_to_print, columns_to_print)
