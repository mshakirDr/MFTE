# Multi-Feature Tagger of English (MFTE)
The MFTE is a Python version of the extended [Multi-Feature Tagger of English (MFTE)](https://github.com/mshakirDr/MultiFeatureTaggerEnglish) based on [Le Foll's (2021) version of MFTE](https://github.com/elenlefoll/MultiFeatureTaggerEnglish) written in Perl. This improved and extended Python version includes semantic tags from Biber (2006) and Biber et al. (1999), including additional tags, e.g., separate tags for third person singular male and female pronouns. This tagger uses `stanza` for grammatical tagging and rule-based regular expressions to tag for multidimensional analysis related features.

# Installation
The software can be used by installing Python and the required packages. Install Python using `anaconda` and then install the required packages. The following are the current dependencies:\
`pip install pandas emoji stanza`\
After installing the dependencies, just download the two Python files `MFTE.py` and `MFTE_gui.py`. You can use the following command to run the GUI version:\
`python "path\to\MFTE\MFTE_gui.py"`\
Or otherwise commandline version can be run as follows with default options:\
`python "path\to\MFTE\MFTE.py" --path "/path/to/corpus/"`\
The GUI version for Windows can be downloaded as a single executable from the following link. There is no need to install anything else.\
[Windows](https://1drv.ms/u/s!AtH0zVEfO5lsgsKxOz4cKq3lOhqIvE8?e=zCOvhq)

# Usage
The usage of the software is straightforward as the screenshot below shows. Just open the folder which contains your text files by clicking on the button. Once you click OK, the software starts with the Part-of-Speech tagging and later with the MFTE tags. Like the original Perl version, the output will be generated in a new folder which will have `_MFTE_tagged` as suffix. This folder contains three subfolders: `MFTE_Tagged`, `StanfordPOS_Tagged` and `Statistics`. The `Statistics` folder is your go to folder which contains the tag counts in different formats: mixed normed (based on 100 nouns or verbs depending on the type of word), raw counts (raw frequencies) and word-based normed (based on 100 words).
![MFTE](https://user-images.githubusercontent.com/46898829/227144641-008478b3-2933-44fb-8e54-b3d848106996.png)

# Tag descriptions
Please refer to `List_Features_MFTE_python_1.4.xlsx` for feature descriptions (work in progress).

# Evaluation
TBD

# Acknowledgements

## Funding
This project has been partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) grant number 452561886.

## Helpful colleagues in the development of MFTE Perl
I would like to thank Peter Uhrig and Michael Franke for supervising my M.Sc. thesis on the development and evaluation of the MFTE. Many thanks to Andrea Nini for releasing the MAT under an open-source licence. Heartfelt thanks also go to Stefanie Evert, Muhammad Shakir, and Luke Tudge who contributed advice and code in various ways (see comments in code for details) and to Larissa Goulart for her insights into the Biber Tagger. Finally, I would also like to thank Dirk Siepmann for supporting this project.

# Citation

## APA
Le Foll, E., & Shakir, M. (2023). MFTE Python (Version 1.0) [Computer software]. https://github.com/mshakirDr/MFTE

## Bibtex

`@software{Le_Foll_MFTE_Python_2023,
author = {Le Foll, Elen and Shakir, Muhammad},
month = {4},
title = {{MFTE Python}},
url = {https://github.com/mshakirDr/MFTE},
version = {1.0},
year = {2023}
}`
# License
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
