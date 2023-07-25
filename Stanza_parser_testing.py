import sys
import os
import inspect 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import stanza
from stanza.pipeline.core import DownloadMethod
from stanza.server import CoreNLPClient
from stanza.utils.conll import CoNLL
import glob
from stanza.models.constituency.tree_reader import read_tree_file

#CUSTOM_PROPS = {"pos.model": "english-bidirectional-distsim-prod1.tagger"}
#There is also a mechanism for only attempting to download models when a particular package is missing. It will reuse an existing resources.json file rather than trying to download it, though.

def extract_phrase_from_constituency (c: str, phrase: str) -> list:
    """Extracts all phrases of a given kind

    Args:
        c (str): constituency tree in brackets format
        phrase (str): phrase to be extracted
    Returns:
        to_return (list): list of phrases
    """
    import re
    to_return = list()
    print(phrase)
    for m in re.finditer(phrase, c):
        print (m.start())
        opening = 0
        closing = 0


#text = r"The solution produced by this process is extraordinarily effective. The time working on this project was awesome. The boy had had his work completed. They are happy and hungry. We went to the museum and watched some artifacts.\
#    The boy who is standing there is my brother."

#nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
#doc = nlp(text)
#print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
text = open(currentdir+"/Constituency_example_sentences.txt").read()
print(text)
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
doc = nlp(text)
for sentence in doc.sentences:
    c: stanza.models.constituency.parse_tree.Tree = sentence.constituency
    print(c.pretty_print())
    #extract_phrase_from_constituency(sentence.constituency, "NP")
    with open(file=currentdir+"/Constituency_example_sentences_tagged.txt", mode="a") as f:
        f.write(str(c)+"\n")
trees = read_tree_file(filename=currentdir+"/Constituency_example_sentences_tagged.txt")
print(trees)