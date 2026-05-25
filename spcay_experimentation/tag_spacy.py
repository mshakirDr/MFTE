import spacy
import os
import glob

def tag_text_files(input_dir, output_directory):
    """
    Reads an English text file, tags it with spaCy (POS tags only),
    and outputs the tagged text to a new file in a specified directory.

    Args:
        input_file_path: Path to the input text file.
        output_directory: Path to the directory where the tagged file will be saved.
    """

    try:
        # Load the English spaCy model (you might need to download it: python -m spacy download en_core_web_sm)
        nlp = spacy.load("en_core_web_trf")
        for input_file_path in glob.glob(input_dir+"*.txt"):
            print(input_file_path)
            # Read the input file
            with open(input_file_path, "r", encoding="utf-8") as infile:
                text = infile.read()

            # Process the text with spaCy
            doc = nlp(text)

            # Create the output directory if it doesn't exist
            os.makedirs(output_directory, exist_ok=True)

            # Construct the output file path
            input_filename = os.path.basename(input_file_path) # Extract filename
            output_file_path = os.path.join(output_directory, input_filename)

            s_list = list()
            for sentence in doc.sents:
                words = []
                for word in sentence:
                    if not word.is_space:
                        words.append(word.text + '_' + word.tag_)
                s_words = " ".join(words)
                s_list.append(s_words)
            s = "\n".join(s_list)
            # Write the tagged text to the output file
            with open(output_file_path, "w", encoding="utf-8") as outfile:
                outfile.write(s)
            print(f"Tagged file saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file_path}")
    except Exception as e:  # Catch other potential errors
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_dir = os.path.dirname(os.getcwd()) + r"\\\MFTE-master\Spacy_eval_files\Texts\\"
    output_dir = os.path.dirname(os.getcwd()) + r"\\\MFTE-master\Spacy_eval_files\Texts_MFTE\POS_Tagged\\"
    print(input_dir)
    tag_text_files(input_dir, output_dir)

