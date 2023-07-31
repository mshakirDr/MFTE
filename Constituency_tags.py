import sys
import os
import inspect 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import stanza
import glob
import re
from stanza.models.constituency.tree_reader import read_tree_file, read_trees
from stanza.models.constituency.parse_tree import Tree
import string

def constituency_to_list_of_words (c: stanza.models.constituency.parse_tree.Tree) -> list:
    """Returns constituency node as a list of words after removing all tags and labels

    Args:
        c (stanza.models.constituency.parse_tree.Tree): constituency node

    Returns:
        sub_list (list): list of words extracted from constituency node
    """
    return re.sub('(\\(\\S*\\s*|\\))', '', str(c)).split(' ')


def find_sub_list_starting_index_in_words_list(words: list, sub_list: list) -> int:
    """Retruns starting index of sub_list in list of words, based on #https://stackoverflow.com/questions/19025655/python-check-next-three-elements-in-list

    Args:
        words (list): a list of words along with tags
        sub_list (list): sub list for which starting index in the main list has to be determined

    Returns:
        index (int): index where the sub list starts in the main list
    """
    words_without_tags = [w.split('_')[0] for w in words]
    for i in range (len(words_without_tags) - len(sub_list) + 1):
        if words_without_tags[i:i + len(sub_list)] == sub_list:
            return i
    return 0

def get_nodes_from_child (tree: stanza.models.constituency.parse_tree.Tree, node: str) -> list:
    """Check if tree has S nodes at the first level, append to to_return and return it

    Args:
        tree (stanza.models.constituency.parse_tree.Tree): Tree to be probed for S nodes

    Returns:
        to_return (list): list of S trees
    """
    to_return = []
    if tree.label == node:
        to_return.append(tree)
    for index, child in enumerate(tree.children):
        to_return.extend(get_nodes_from_child(child, node))
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
    for tree in trees:
        nodes_list.extend(get_nodes_from_child(tree, node))
    return nodes_list

def tag_non_finite_relative_clauses(words: list, trees: list) -> list:
    """Return words list after adding VBNRel, VBGRel tags

    Args:
        words (list): list of words that is previously tagged
        trees (list): list of trees tagged by stanza

    Returns:
        words (list): list of words after adding VBNRel, VBGRel tags
    """
    np_trees = get_nodes_of_interest(trees, 'NP') #get all NP nodes
    for np_tree in np_trees:
        #present participial relative clauses (NP (NP xxx) (VP (VBG xxxx)))
        if len(np_tree.children) > 1:
            if np_tree.children[0].label == 'NP' and np_tree.children[1].label == 'VP': #first child is NP and 2nd child a VP
                if (np_tree.children[1].children[0].label == 'VBG'): #first word in the VP is VBG
                    #print(np_tree)
                    np_tree_list_of_words = constituency_to_list_of_words(np_tree.children[1])
                    index = find_sub_list_starting_index_in_words_list(words, np_tree_list_of_words)
                    words[index] = re.sub("_(\w+)", "_VBGRel", words[index])
        
        #past participial relative clauses (NP (NP xxx) (VP (VBN xxxx)))
        if len(np_tree.children) > 1:
            if np_tree.children[0].label == 'NP' and np_tree.children[1].label == 'VP': #first child is an NP and 2nd child is a VP
                if np_tree.children[1].children[0].label == 'VBN': #first word in the VP is VBN
                    #print(np_tree)
                    np_tree_list_of_words = constituency_to_list_of_words(np_tree.children[1])
                    index = find_sub_list_starting_index_in_words_list(words, np_tree_list_of_words)
                    words[index] = re.sub("_(\w+)", "_VBNRel", words[index])

                elif ((np_tree.children[1].children[0].label == 'ADVP' and  np_tree.children[1].children[1].label == 'VBN')): #first child in VP is ADVP, 2nd is VBN
                    np_tree_list_of_words = constituency_to_list_of_words(np_tree.children[1])
                    index = find_sub_list_starting_index_in_words_list(words, np_tree_list_of_words)
                    words[index+1] = re.sub("_(\w+)", "_VBNRel", words[index+1]) #+1 due to the presence of adverb before the verb                   

                if len(np_tree.children[1].children) > 2: #check for two VPs joined by and
                    if (np_tree.children[1].children[0].label == 'VP' and 
                    np_tree.children[1].children[1].label == 'CC' and 
                    np_tree.children[1].children[2].label == 'VP'):
                        if np_tree.children[1].children[0].children[0].label == 'VBN': #first word in the VP is VBN
                            #print(np_tree.children[1].children[0])
                            np_tree_list_of_words = constituency_to_list_of_words(np_tree.children[1].children[0])
                            #print(np_tree_list_of_words)
                            index = find_sub_list_starting_index_in_words_list(words, np_tree_list_of_words)
                            words[index] = re.sub("_(\w+)", "_VBNRel", words[index])
                        
                        if np_tree.children[1].children[2].children[0].label == 'VBN': #first word in the VP is VBN
                            #print(np_tree.children[1].children[2])
                            np_tree_list_of_words = constituency_to_list_of_words(np_tree.children[1].children[2])
                            #print(np_tree_list_of_words)
                            index = find_sub_list_starting_index_in_words_list(words, np_tree_list_of_words)
                            words[index] = re.sub("_(\w+)", "_VBNRel", words[index])

        #past participial relative clauses with additional items like punctuation (NP (NP xxx) (, ,) (VP (VBN xxxx))) occuring after a comma
        if len(np_tree.children) > 2:
            if np_tree.children[0].label == 'NP' and np_tree.children[1].label in string.punctuation and np_tree.children[2].label == 'VP': #first child is NP, 2nd child is punctuation and 3rd child a VP
                if (np_tree.children[2].children[0].label == 'ADVP' and  np_tree.children[2].children[1].label == 'VBN'): #first child in VP is ADVP, 2nd is VBN
                    np_tree_list_of_words = constituency_to_list_of_words(np_tree.children[2])
                    index = find_sub_list_starting_index_in_words_list(words, np_tree_list_of_words)
                    words[index+1] = re.sub("_(\w+)", "_VBNRel", words[index+1]) #+1 due to the presence of adverb before the verb
                
                elif (np_tree.children[2].children[0].label == 'VBN'): ##first word in the VP is VBN
                    np_tree_list_of_words = constituency_to_list_of_words(np_tree.children[2])
                    index = find_sub_list_starting_index_in_words_list(words, np_tree_list_of_words)
                    words[index] = re.sub("_(\w+)", "_VBNRel", words[index])

        #VBG as attributive adjectives (NP (DT the) (JJ Indian) (VBG founding) (NNS fathers))
        labels = [c.label for c in np_tree.children]
        if any(label == 'VBG' for label in labels) and not any(label == 'VP' for label in labels):
            np_tree_list_of_words = constituency_to_list_of_words(np_tree)
            item_index = labels.index('VBG')
            sub_list_index = find_sub_list_starting_index_in_words_list(words, np_tree_list_of_words)
            words[sub_list_index+item_index] = re.sub("_(\w+)", "_JJAT JJATother", words[sub_list_index+item_index])

        #VBN as attributive adjectives (NP (VBN perceived) (JJ diplomatic) (NN affront))
        labels = [c.label for c in np_tree.children]
        if any(label == 'VBN' for label in labels) and not any(label == 'VP' for label in labels):
            np_tree_list_of_words = constituency_to_list_of_words(np_tree)
            item_index = labels.index('VBN')
            sub_list_index = find_sub_list_starting_index_in_words_list(words, np_tree_list_of_words)
            words[sub_list_index+item_index] = re.sub("_(\w+)", "_JJAT JJATother", words[sub_list_index+item_index])

    return words

def tag_non_finite_participial_clauses(words: list, trees: list) -> list:
    """Return words list after adding VBNCls, VBGCls tags

    Args:
        words (list): list of words that is previously tagged
        trees (list): list of trees tagged by stanza

    Returns:
        words (list): list of words after adding VBGCls, VBNCls tags
    """
    s_trees = get_nodes_of_interest(trees, 'S') #get all sentence nodes
    for s_tree in s_trees:
        #present participial clauses (S (VP (VBG xxxx)))
        if s_tree.children[0].label == 'VP': #first child is a VP
            if s_tree.children[0].children[0].label == 'VBG': #first word in the VP is VBG
                #print(s_tree)
                s_tree_list_of_words = constituency_to_list_of_words(s_tree)
                index = find_sub_list_starting_index_in_words_list(words, s_tree_list_of_words)
                words[index] = re.sub("_(\w+)", "_VBGCls", words[index])
                #print(words[index])

        #past participial clauses (S (VP (VBN xxxx)))
        if s_tree.children[0].label == 'VP': #first child is a VP
            if s_tree.children[0].children[0].label == 'VBN': #first word in the VP is VBN
                s_tree_list_of_words = constituency_to_list_of_words(s_tree)
                index = find_sub_list_starting_index_in_words_list(words, s_tree_list_of_words)
                words[index] = re.sub("_(\w+)", "_VBNCls", words[index])
    return words        

def tag_constituency (words: list, pos_tagged_file_path: str) -> list:
    """Returns words list after adding constituency based tags VBGCls, VBNCls, VBGRel, VBNRel etc.

    Args:
        words (list): list of words after simple and extended tags have been added
        pos_tagged_file_path (str): pos tagged file path to retrieve corresponding constituency tagged file path

    Returns:
        word (list): list of words after added constituency based tags
    """
    input_file_path = pos_tagged_file_path.replace('POS_Tagged', 'Constituency_Trees') #get corresponding constituency tree file path
    trees_text = open(file=input_file_path, mode='r', encoding='UTF-8', errors='ignore').read()
    trees = read_trees(trees_text)
    words = tag_non_finite_participial_clauses(words, trees)
    words = tag_non_finite_relative_clauses(words, trees)
    return words

if __name__ == "__main__":
    words = open(file=r"D:\PostDoc\ExtraAcademicWork\MFTE\Test\Corpus_MFTE_tagged\MFTE_Tagged\abc.txt", mode='r', encoding='UTF-8', errors='ignore').read().splitlines()
    pos_file_path = r"D:\PostDoc\ExtraAcademicWork\MFTE\Test\Corpus_MFTE_tagged\POS_Tagged\abc.txt"
    tag_constituency(words, pos_file_path)