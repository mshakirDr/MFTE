# Multi-Feature Tagger of English (MFTE)
The Multi-Feature Tagger of English (MFTE) was originally based on the [MFTE Perl (Le Foll 2021)](https://github.com/elenlefoll/MultiFeatureTaggerEnglish). The present, substantially improved Python version considerably expands the number of tagged features. In addition to many new lexico-grammatical features, it includes the semantic tags from Biber (2006) and Biber et al. (1999). The MFTE Python relies on the Python NLP library `stanza` for grammatical part-of-speech tagging before applying rule-based regular expressions to tag for a range of more complex lexico-grammatical and semantic features typically used in multidimensional analysis (MDA; Biber 1984; 1988; 1995).


# Installation

## Standalone executeable (GUI) for Windows (updated 1-06-2023; out of date, the below installation method is highly recommended)
The GUI version for Windows can be downloaded as a single executable from the following link:

[GUI version for Windows](https://1drv.ms/u/s!AtH0zVEfO5lsguLldNW2aRyzM8Gxia8?e=zKghN1).

There is no need to install anything else.

## Command-line installation using Anaconda
To use this software, you must first install Python. We recommend that you install Python using `anaconda` ([video tutorial for Windows](https://www.youtube.com/watch?v=UTqOXwAi1pE), [video tutorial for Mac](https://www.youtube.com/watch?v=n83J8cBytus)). Then, install the MFTE using the following command on Anaconda Terminal in Windows, Mac, or Linux:

`pip install MFTE`

Afterwards you can run `MFTE` from the same terminal window: `MFTE` for command-line and `MFTE_gui` for the GUI (graphical user interface) version.

#### Optional installation of GPU support (especially useful for large corpora)
If you have an nVidia GPU in your system (Windows or Linux), you can install GPU support by installing CUDA enabled [`Pytorch`](https://pytorch.org/get-started/locally/) (the framework `stanza` uses under the hood):

For Windows:
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

For Linux:
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

For MacOS 12.3+ MPS acceleration is available through nightly builds. Run the following command to update `pytorch`:
`pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu` 

# Usage 
## Usage of the GUI version
To launch the GUI version on Windows, Linux or MacOS, simply call up `MFTE_gui` in the Anaconda terminal.

The usage of the MFTE using the GUI is straightforward as the screenshot below shows. Additional information about each option is available in tooltips. Simply hover your mouse over a checkbox or button to find out more about each option. The MFTE Python includes a simple and an extended tagset. By default, the extended tagset is used (see feature descriptions). By default, type-token-ratios (TTR) are calculated on the basis of the first 400 words of each text. This number should not be fewer than the shortest text in your corpus if you wish to use this feature in your analyses. 

![MFTE](https://user-images.githubusercontent.com/46898829/227144641-008478b3-2933-44fb-8e54-b3d848106996.png)

Simply check or uncheck the options you require and then open the folder which contains your text files by clicking on the `Select corpus directory` button. As soon as you click OK, the software begins with the part-of-speech (POS) tagging and later with the MFTE tags. The output is generated in a new folder which preserves the name of the original folder complemented by `_MFTE` as a suffix (see Outputs). 

# Usage of the command-line version
The `mfte` script takes the following optional arguments, which you can change as you like:

|Argument|Explanation|
|---------|---------|
|`--path 'path\to\corpus'`|path to the text files folder|
|`--ttr 400`| By default, type-token-ratios (TTR) are calculated on the basis of the first 400 words of each text. So default is `400`|
|`--extended True`| The MFTE Python includes a simple and an extended tagset so use `True` or `False`; by default it is enabled using `True`|
|`--parallel_md_tagging False`| enable MD tagging of multiple files at the same time (high CPU usage) `True` or `False`; default is `False`|

The complete command will look like this:

`mfte --path "/path/to/corpus/" --ttr 400 --extended True --parallel_md_tagging False`

# Feature descriptions
The MFTE Python tags over 100 lexico-grammatical and semantic features. Please refer to the [`List_Features_MFTE_python_1.0.0.pdf`](https://github.com/mshakirDr/MFTE/blob/master/List_Features_MFTE_python_1.0.0.pdf).

Further information can be be found in [Introducing the MFTE Perl](https://github.com/elenlefoll/MultiFeatureTaggerEnglish/blob/main/Introducing_the_MFTE_v3.0.pdf), a 50-page document based on revised, selected chapters from an M.Sc. thesis submitted for the degree of Master of Science in Cognitive Science at the Institute of Cognitive Science, Osnabrück University (Germany) in November 2021. It outlines the steps involved in the development of the Perl MFTE. Section 2.1 outlines its specifications, which were drawn up on the basis of the features needed to carry out MDA and taking account of the advantages and limitations of existing taggers. The following sections explain the methodological decisions involved in the selection of the features to be identified by the MFTE (2.2), the details of the regular expressions used to identify these features (2.3) and the procedure for normalising the feature counts (2.4). Section 2.5 describes the outputs of the tagger. Chapter 3 presents the method and results of an evaluation of the accuracy of the MFTE. It reports the results of comparisons of the tags assigned by the MFTE and by two human annotators to calculate precision and recall rates for each linguistic feature across a range of contrasting text registers. The data and code used to analyse the evaluation results are also available in the corresponding [GitHub repository](https://github.com/elenlefoll/MultiFeatureTaggerEnglish).

# Outputs
The  `[prefix]_MFTE` output folder contains three subfolders: `MFTE_Tagged`, `POS_Tagged` and `Statistics`. The first two folders contain the tagged texts with which you can check the accuracy of the tagging process. The `Statistics` folder is your go-to folder to further analyses. It contains feature counts in the form of comma-separated-values files (`.csv`). Each row corresponds to a text file from the corpus tagged and each column corresponds to a linguistic feature. The MFTE outputs three different tables of feature counts:
1.	```counts_mixed_normed.csv```            Normalised feature frequencies calculated on the basis of linguistically meaningful normalisation baselines (as listed in the sixth column of [`List_Features_MFTE_python_1.0.0.pdf`](https://github.com/mshakirDr/MFTE/blob/master/List_Features_MFTE_python_1.0.0.pdf), see also Section 5.3.4 in Le Foll 2024)
2.	```counts_word-based_normed.csv```            Feature frequencies normalised to 100 words
3.	```counts_raw.csv```                         Raw (unnormalised) feature counts

Note that the MFTE only tags and computes count tallies and relative frequencies of all the features. It does not compute perform the multidimensional analysis itself. R scripts to carry out MDA analysis using PCA on the basis of the outputs of the MFTE can be found in the [online supplements](https://elenlefoll.github.io/TextbookMDA/) to Le Foll (2024).

# Evaluation

Le Foll, Elen & Muhammad Shakir. 2025. The Multi-Feature Tagger of English (MFTE): Rationale, Description and Evaluation. Research in Corpus Linguistics 13(2). 63–93. https://doi.org/10.32714/ricl.13.02.03.

# Acknowledgements

## Funding
This project has been partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) grant number 452561886.

## Acknowledgements from the MFTE Perl
Elen would like to extend special thanks to Peter Uhrig and Michael Franke for supervising her M.Sc. thesis on the development and evaluation of [the first, Perl version of the MFTE](https://github.com/elenlefoll/MultiFeatureTaggerEnglish). Many thanks to Andrea Nini for releasing the [MAT](http://sites.google.com/site/multidimensionaltagger) under an open-source licence, which served as the baseline for this previous version of the MFTE. Heartfelt thanks also go to Stefanie Evert and Luke Tudge who contributed advice and code in various ways and to Larissa Goulart for her insights into the Biber Tagger. Finally, Elen would also like to thank Dirk Siepmann for supporting this project.

# Citation
Please cite the MFTE as follows:

### APA
Le Foll, E., & Shakir, M. (2024). The Multi-Feature Tagger of English (MFTE): Rationale, description and evaluation. Research in Corpus Linguistics, 13(2), 63–93. https://doi.org/10.32714/ricl.13.02.03

Le Foll, E., & Shakir, M. (2023). MFTE Python (Version 1.0) [Computer software]. https://github.com/mshakirDr/MFTE

### Bibtex
`@article{lefollMultiFeatureTaggerEnglish2025,
	title = {The {Multi}-{Feature} {Tagger} of {English} ({MFTE}): {Rationale}, {Description} and {Evaluation}},
	volume = {13},
	doi = {https://doi.org/10.32714/ricl.13.02.03},
	number = {2},
	journal = {Research in Corpus Linguistics},
	author = {Le Foll, Elen and Shakir, Muhammad},
	year = {2025},
	pages = {63--93},
}`

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

Le Foll, Elen. 2024. Textbook English: A Multi-Dimensional Approach (Studies in Corpus Linguistics 116). Amsterdam: John Benjamins. https://doi.org/10.1075/scl.116. Open Access version: https://osf.io/yhxft. Online supplements: https://elenlefoll.github.io/TextbookMDA/.

Nini, Andrea (2014). Multidimensional Analysis Tagger (MAT). https://sites.google.com/site/multidimensionaltagger.

Nini, Andrea (2019). The Muli-Dimensional Analysis Tagger. In Berber Sardinha, T. & Veirano Pinto M. (eds), Multi-Dimensional Analysis: Research Methods and Current Issues, 67-94, London; New York: Bloomsbury Academic.

Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. https://doi.org/10.48550/ARXIV.2003.07082
