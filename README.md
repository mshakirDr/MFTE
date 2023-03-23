# MFTE
MFTE is the Python version of the extended MFTE [Multi-Feature Tagger of English (MFTE)](https://github.com/mshakirDr/MultiFeatureTaggerEnglish) based on Le Foll's version of [MFTE](https://github.com/elenlefoll/MultiFeatureTaggerEnglish) written in Perl. The extended version includes semantic tags from Biber (2006) and Biber et al. (1999), including other specific tags, for example separate tags for third person singular male and female pronouns. This tagger uses `stanza` for grammatical tagging and rule-based regular expressions to tag for the multidimensional analysis related features.

# Installation
The software can be used by installing Python and the required packages. Install Python using `anaconda` and then install the required packages. The following are the current dependencies:\
`pip install pandas emoji stanza`\
After installing the dependencies, just download the two Python files `MFTE.py` and `MFTE_gui.py`. You can use the following command to run the GUI version:\
`python path\to\MFTE\MFTE_gui.py`\
The GUI version for Windows can be downloaded as a single executable from the following link. There is no need to install anything else.
[Windows](https://1drv.ms/u/s!AtH0zVEfO5lsgsKxOz4cKq3lOhqIvE8?e=zCOvhq)

# Usage
The usage of the software is straightforward as the screenshot below shows. Just open the folder which contains your text files by clicking on the button. Once you click OK, the software starts with grammatical tagging and later with MFTE tags. Like the original version, the output will be generated in a new folder which will have `_MFTE_tagged` as a suffix. There are three subfolders: `MFTE_Tagged`, `StanfordPOS_Tagged` and `Statistics`. The `Statistics` folder is your go to folder which contains the tag counts in different formats: mixed normed (based on 100 nouns or verbs depending on the type of word): raw counts (raw frequencies) and word-based normed (based on 100 words).
![MFTE](https://user-images.githubusercontent.com/46898829/227144641-008478b3-2933-44fb-8e54-b3d848106996.png)

# Tag descriptions
Please refer to `List_Features_MFTE_python_1.4.xlsx` for feature descriptions.

# Evaluation
TBD

# Citation
