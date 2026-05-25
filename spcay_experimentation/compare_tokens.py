import re
import pandas as pd
import os
import glob

def text_file_to_list(file_path):
    l = open(file=file_path, mode="r", encoding="UTF-8").readlines()
    return l

def read_tokens_from_excel(file_path):
    """
    Read tokens from Excel file's column B starting from row 2.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        str: Concatenated tokens with newlines
    """
    df = pd.read_excel(file_path, header=0)
    #print(df.columns)
    l = df["Output"].to_list()
    return df

    


if __name__ == "__main__":
    eval_dir = os.path.dirname(os.getcwd()) + r"\\MFTE\\evaluation\\"
    spacy_tagged_dir = os.path.dirname(os.getcwd()) + r"\\Spacy_eval_files\Texts_MFTE\MFTE_Tagged\\"
    excel_files = glob.glob(eval_dir+"*.xlsx")
    for excel_file in excel_files:
        file_name = os.path.basename(excel_file).split('.')[0] + ".txt"
        print(file_name)
        tokens_vertical = read_tokens_from_excel(excel_file)
        tokens_spacy = text_file_to_list(spacy_tagged_dir+file_name)
        print(len(tokens_vertical), len(tokens_spacy))
        if len(tokens_vertical) == len(tokens_spacy):
            print(file_name)
