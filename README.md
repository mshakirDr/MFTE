# Multi-Feature Tagger of English (MFTE)
The MFTE is a Python version of the extended [Multi-Feature Tagger of English (MFTE)](https://github.com/mshakirDr/MultiFeatureTaggerEnglish) based on [Le Foll's (2021) version of the MFTE](https://github.com/elenlefoll/MultiFeatureTaggerEnglish) written in Perl. This improved and extended Python version includes semantic tags from Biber (2006) and Biber et al. (1999), as well as additional tags, e.g., separate tags for third person singular male and female pronouns. This tagger first uses the Python NLP library `stanza` for grammatical part-of-speech tagging before applying rule-based regular expressions to tag for a range of more complex lexico-grammatical and semantic features typically used in multidimensional analysis (MDA; cf. Biber 1984; 1988).

# Current version
## 1.6 (January 2024)
- Added six new tags (`VBNCls, VBNRel, VBGCls, VBGRel, CCCls, CCPhrs`) that are assigned using consituency parsing function of `stanza`. GUI and commandline files updated
- Resume ability (skip files that are already tagged by `stanza` and `MFTE`.)

# Installation
## Command line using Anaconda
This software can be used by installing Python and the required packages. We recommend that you install Python using `anaconda` ([video tutorial for Windows](https://www.youtube.com/watch?v=UTqOXwAi1pE), [video tutorial for Mac](https://www.youtube.com/watch?v=n83J8cBytus)) and then install the required packages. Just copy and paste the following command in Windows or Mac Terminal to install the current dependencies:

`pip install pandas emoji stanza`

After installing the dependencies, simply download the two Python files `MFTE.py` and `MFTE_gui.py`. You can use the following command to run the GUI version (see below for details):

`python "path\to\MFTE\MFTE_gui.py"`

Or otherwise the command-line version can be run as follows with default options:

`python "path\to\MFTE\MFTE.py" --path "/path/to/corpus/"`

The script takes the following optional arguments so you can change them as you like:

|Argument|Explanation|
|---------|---------|
|`--path 'path\to\corpus'`|path to the text files folder|
|`--ttr 400`| By default, type-token-ratios (TTR) are calculated on the basis of the first 400 words of each text. So default is `400`|
|`--extended True`| The MFTE Python includes a simple and an extended tagset so use `True` or `False`; by default it is enabled using `True`|
|`--parallel_md_tagging False`| enable MD tagging of multiple files at the same time (high CPU usage) `True` or `False`; default is `False`|
|`--constituency_tagging False`| enable constituency tree based additional tags (requires nVidia GPU otherwise CPU processing time icreases significantly) `True` or `False`; default is `False`|

The complete command will look like this:

`python "path\to\MFTE\MFTE.py" --path "/path/to/corpus/" --ttr 400 --extended True --parallel_md_tagging False --constituency_tagging False`
## Standalone executeable (GUI)
### Windows (update 1-06-2023)
The GUI version for Windows can be downloaded as a single executable from the following link:

[GUI version for Windows](https://1drv.ms/u/s!AtH0zVEfO5lsguLldNW2aRyzM8Gxia8?e=zKghN1).

There is no need to install anything else. Additional information about each option is available in tooltips. Simply hover your mouse over a checkbox or button to find out more about each option.

The usage of the software using the GUI is straightforward as the screenshot below shows. Simply open the folder which contains your text files by clicking on the `Select corpus directory` button. Once you click OK, the software begins with the part-of-speech (POS) tagging and later with the MFTE tags. As in the original Perl version, the output is generated in a new folder which preserves the name of the original folder complemented by `_MFTE` as a suffix. 

![MFTE](https://user-images.githubusercontent.com/46898829/227144641-008478b3-2933-44fb-8e54-b3d848106996.png)

By default, type-token-ratios (TTR) are calculated on the basis of the first 400 words of each text.

The MFTE Python includes a simple and an extended tagset. By default, the extended tagset is used (see feature descriptions).

# Feature descriptions
The MFTE Python tags over 100 lexico-grammatical and semantic features. Please refer to the [`List_Features_MFTE_python_1.5.xlsx`](https://github.com/mshakirDr/MFTE/blob/master/List_Features_MFTE_python_1.5.xlsx) and the [Wiki](https://github.com/mshakirDr/MFTE/wiki) for details (work in progress).

Further information can be be found in [Introducing the MFTE Perl](https://github.com/elenlefoll/MultiFeatureTaggerEnglish/blob/main/Introducing_the_MFTE_v3.0.pdf), a 50-page document based on revised, selected chapters from an M.Sc. thesis submitted for the degree of Master of Science in Cognitive Science at the Institute of Cognitive Science, Osnabrück University (Germany) in November 2021. It outlines the steps involved in the development of the Perl MFTE. Section 2.1 outlines its specifications, which were drawn up on the basis of the features needed to carry out MDA and taking account of the advantages and limitations of existing taggers. The following sections explain the methodological decisions involved in the selection of the features to be identified by the MFTE (2.2), the details of the regular expressions used to identify these features (2.3) and the procedure for normalising the feature counts (2.4). Section 2.5 describes the outputs of the tagger. Chapter 3 presents the method and results of an evaluation of the accuracy of the MFTE. It reports the results of comparisons of the tags assigned by the MFTE and by two human annotators to calculate precision and recall rates for each linguistic feature across a range of contrasting text registers. The data and code used to analyse the evaluation results are also available in the corresponding [GitHub repository](https://github.com/elenlefoll/MultiFeatureTaggerEnglish).

# Outputs

The  `[prefix]_MFTE` output folder contains three subfolders: `MFTE_Tagged`, `POS_Tagged` and `Statistics`. The first two folders contain the tagged texts with which you can check the accuracy of the tagging process. The `Statistics` folder is your go-to folder to further analyses. It contains feature counts in the form of comma-separated-values files (`.csv`). Each row corresponds to a text file from the corpus tagged and each column corresponds to a linguistic feature. The MFTE outputs three different tables of feature counts:
1.	```counts_mixed_normed.csv```            Normalised feature frequencies calculated on the basis of linguistically meaningful normalisation baselines (as listed in the fifth column of the `List_Features_MFTE_python_1.5.xlsx`)
2.	```counts_word-based_normed.csv```            Feature frequencies normalised to 100 words
3.	```counts_raw.csv```                         Raw (unnormalised) feature counts

Note that the MFTE only tags and computes count tallies and relative frequencies of all the features. It does not compute perform the multidimensional analysis itself. R scripts to carry out MDA analysis using EFA and PCA on the basis of the outputs of the MFTE will soon be added to this repository.

# Evaluation
TBD

# Acknowledgements

## Funding
This project has been partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) grant number 452561886.

## Acknowledgements from the MFTE Perl
Elen would like to extend special thanks to Peter Uhrig and Michael Franke for supervising her M.Sc. thesis on the development and evaluation of [the first, Perl version of the MFTE](https://github.com/elenlefoll/MultiFeatureTaggerEnglish). Many thanks to Andrea Nini for releasing the [MAT](http://sites.google.com/site/multidimensionaltagger) under an open-source licence, which served as the baseline for this previous version of the MFTE. Heartfelt thanks also go to Stefanie Evert and Luke Tudge who contributed advice and code in various ways and to Larissa Goulart for her insights into the Biber Tagger. Finally, Elen would also like to thank Dirk Siepmann for supporting this project.

# Citation

The MFTE can be cited as follows:

### APA

Le Foll, E., & Shakir, M. (2023). MFTE Python (Version 1.0) [Computer software]. https://github.com/mshakirDr/MFTE

### Bibtex

`@software{Le_Foll_MFTE_Python_2023,
author = {Le Foll, Elen and Shakir, Muhammad},
month = {4},
title = {{MFTE Python}},
url = {https://github.com/mshakirDr/MFTE},
version = {1.0},
year = {2023}
}`

Please also cite the Python library Stanza.

### APA

Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. https://doi.org/10.48550/ARXIV.2003.07082

### Bibtex
`@article{qi_stanza_2020,
	title = {Stanza: A Python Natural Language Processing Toolkit for Many Human Languages},
	doi = {10.48550/ARXIV.2003.07082},
	shorttitle = {Stanza},
	author = {Qi, Peng and Zhang, Yuhao and Zhang, Yuhui and Bolton, Jason and Manning, Christopher D.},
	urldate = {2023-04-04},
	date = {2020},
}`


# License
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# References
Biber, Douglas (1984). A model of textual relations within the written and spoken modes. University of Southern California. Unpublished PhD thesis.

Biber, Douglas (1988). Variation across speech and writing. Cambridge: Cambridge University Press.

Biber, Douglas (1995). Dimensions of Register Variation. Cambridge, UK: Cambridge University Press.

Biber, D. (2006). University language: A corpus-based study of spoken and written registers. Benjamins.

Biber, D., Johansson, S., Leech, G., Conrad, S., & Finegan, E. (1999). Longman Grammar of Spoken and Written English. Longman Publications Group.

Conrad, Susan & Douglas Biber (eds.) (2013). Variation in English: Multi-Dimensional Studies (Studies in Language and Linguistics). New York: Routledge.

Le Foll, Elen (2021). A New Tagger for the Multi-Dimensional Analysis of Register Variation in English. Osnabrück University: Institute of Cognitive Science Unpublished M.Sc. thesis.

Nini, Andrea (2014). Multidimensional Analysis Tagger (MAT). https://sites.google.com/site/multidimensionaltagger.

Nini, Andrea (2019). The Muli-Dimensional Analysis Tagger. In Berber Sardinha, T. & Veirano Pinto M. (eds), Multi-Dimensional Analysis: Research Methods and Current Issues, 67-94, London; New York: Bloomsbury Academic.

Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. https://doi.org/10.48550/ARXIV.2003.07082
