import sys
import os
import inspect 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import stanza
from stanza.pipeline.core import DownloadMethod
from stanza.server import CoreNLPClient
import glob

#CUSTOM_PROPS = {"pos.model": "english-bidirectional-distsim-prod1.tagger"}
#There is also a mechanism for only attempting to download models when a particular package is missing. It will reuse an existing resources.json file rather than trying to download it, though.



text = r"The solution produced by this process is extraordinarily effective. The time working on this project was awesome. The boy had had his work completed. They are happy and hungry. We went to the museum and watched some artifacts.\
    The boy who is standing there is my brother."
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
doc = nlp(text)
print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
doc = nlp(text)
for sentence in doc.sentences:
    print(sentence.constituency)
    tree = sentence.constituency
    print(tree.children)
    print(tree.children[0].children)
