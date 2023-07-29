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



#text = r"The solution produced by this process is extraordinarily effective. The time working on this project was awesome. The boy had had his work completed. They are happy and hungry. We went to the museum and watched some artifacts.\
#    The boy who is standing there is my brother."

#nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
#doc = nlp(text)
#print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
text = open(currentdir+"/Constituency_example_sentences.txt").read()
# print(text)
# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
# doc = nlp(text)
# for sentence in doc.sentences:
#     c: stanza.models.constituency.parse_tree.Tree = sentence.constituency
#     print(c.pretty_print())
#     #extract_phrase_from_constituency(sentence.constituency, "NP")
#     with open(file=currentdir+"/Constituency_example_sentences_tagged.txt", mode="a") as f:
#         f.write(str(c)+"\n")
trees = read_tree_file(filename=currentdir+"/Constituency_example_sentences_tagged.txt")

def get_nodes_from_child (tree: stanza.models.constituency.parse_tree.Tree, node: str) -> list:
    """Check if tree has S nodes at the first level, append to to_return and return it

    Args:
        tree (stanza.models.constituency.parse_tree.Tree): Tree to be probed for S nodes

    Returns:
        to_return (list): list of S trees
    """
    to_return = []
    for index, child in enumerate(tree.children):
        if child.label == node:
            to_return.append(child) 
    return to_return

def get_nodes_of_interest(trees: list, node: str) -> list:
    """Retruns a list of nodes according to provided citeria, e.g. S nodes

    Args:
        trees (list): list of trees to process
        node (str): list of nodes extracted from trees

    Returns:
        nodes_list (list): list of trees having the given 'node' as head
    """
    nodes_list = []
    nodes_list_str = [] #make sure no duplicates, string list because node_list will be used for further comparisons later
    children = []
    for tree in trees:
        #start node list string a new for each tree (at this level it should be a sentence tree)
        nodes_list_str = [str(n) for n in nodes_list]
        nodes_list_temp = get_nodes_from_child(tree, node)
        nodes_list = nodes_list + [n for n in nodes_list_temp if str(n) not in nodes_list_str]
        nodes_list_str = nodes_list_str + [str(n) for n in nodes_list_temp if str(n) not in nodes_list_str]
        children = children + [c for c in tree.children]
        for index, child in enumerate(children):
            nodes_list_temp = get_nodes_from_child(child, node)
            nodes_list = nodes_list + [n for n in nodes_list_temp if str(n) not in nodes_list_str]
            nodes_list_str = nodes_list_str + [str(n) for n in nodes_list_temp if str(n) not in nodes_list_str]
            if len(child.children) > 0:
                children = children + [c for c in child.children]
    return nodes_list
S = get_nodes_of_interest(trees, 'S')
[print(str(s) + '\n') for s in S]
NPs = get_nodes_of_interest(trees, 'NP')
[print(str(s) + '\n') for s in NPs]
NP = NPs.pop()
print('_________________')
for c in NP.children:
    print(c[0])

