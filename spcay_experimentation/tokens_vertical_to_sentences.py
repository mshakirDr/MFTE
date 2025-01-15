import re
import pandas as pd
import os
import glob

def detokenize_text(text):
    """
    Convert vertically tokenized text back to normal sentence flow.
    
    Args:
        text (str): Input text with tokens separated by newlines
        
    Returns:
        str: Detokenized text with proper spacing and punctuation
    """
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Remove spaces before punctuation marks
    punctuation = r'[\.,!?;:\'"\)\]}]'
    text = re.sub(f'\s+({punctuation})', r'\1', text)
    
    # Fix spacing after opening brackets/quotes
    text = re.sub(r'([\(\[{"\'])\s+', r'\1', text)
    
    return text

def read_tokens_from_excel(file_path):
    """
    Read tokens from Excel file's column B starting from row 2.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        str: Concatenated tokens with newlines
    """
    try:
        # Read Excel file, skipping the first row
        df = pd.read_excel(file_path, usecols=[1], skiprows=1, header=None)
        
        # Filter out any empty cells and convert to list
        tokens = df[df[1].notna()][1].tolist()
        
        # Join tokens with newlines
        return '\n'.join(str(token) for token in tokens)
        
    except Exception as e:
        raise Exception(f"Error reading Excel file: {str(e)}")
    


if __name__ == "__main__":
    input_dir = os.path.dirname(os.getcwd()) + r"\\MFTE\\evaluation\\"
    output_dir = os.path.dirname(os.getcwd()) + r"\\MFTE\\spcay_experimentation\\"
    print(input_dir)
    excel_files = glob.glob(input_dir+"*.xlsx")
    for excel_file in excel_files:
        file_name = os.path.basename(excel_file).split('.')[0] + ".txt"
        print(file_name)
        tokens_vertical = read_tokens_from_excel(excel_file)
        tokens_h = detokenize_text(tokens_vertical)
        with open(file=output_dir+file_name, mode="w", encoding="UTF-8") as f:
            f.write(tokens_h)