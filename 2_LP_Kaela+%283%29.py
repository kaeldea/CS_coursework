
# coding: utf-8

# ## Assignment 2
# 
# ### Part 1 (FSAs)

# 1.- Define a deterministic finite-state automaton that accepts strings that have an odd number of 0’s and any number of 1’s.
# 

# M = (Q, Σ, δ, q0, F) 
# 
# Q = {q1, q2},
# 
# Σ = {0, 1},
# 
# q0 = q0,
# 
# F = {q1}
# 
# ![IMG_8956%20%281%29.JPG](attachment:IMG_8956%20%281%29.JPG)

# 2.- Implement a regular expression stemmer that can process the following text. 
# 
# *Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.*

# In[26]:

import nltk
import re

raw = "Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma."
def stem(word):
    regexpression = r'^(.*?)(tion|al|ic|ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexpression, word)[0] #find all method finds all (non-overlapping) matches of the given regular expression
    return stem

tokens = nltk.word_tokenize(raw)

print("Results of regular expression stemmer:")
print([stem(t) for t in tokens]) #list comprehension 
print("")


# 3.- Expand the grammar grammar1.cfg so that it also parses the sentence
# 
# *John said to Bob that Mary saw a man with a telescope*
# 

# In[27]:

import nltk

cp = nltk.parse.load_parser('grammar2.cfg') #expanded grammar1.cfg in a text editor - showed some notes below 

sent = "John said to Bob that Mary saw a man with a telescope"

tokens = sent.split()

trees = cp.parse(tokens)

for tree in trees: 
    print(tree)
 
    
#Mary = NP
#VP
#saw a man = V NP (V Det N )  
#with a Telescope = PP ((P NP) (Det N))
#John = NP 
#To Bob = NP
#Said to Bob = VP (V NP)
#John said to Bob = S (NP VP)
#a man = NP (Det N)
#with a telescope = PP 


# ### Part 2 (Wordnet)

# In this assignment you will be creating a program for learning languages! We will print a fairy tale, and propose a simple test to check if the language learner knows the target language!
# 
# There is a file called `little-red-riding-hood-clean-5lines.txt`, which, as the name suggests, contains the story *Little Red Riding Hood*.
# 
# Your job is to do the following:

# #### 1st step:
# 
#  * Open and load file
#  * Read text and remove punctuation (Remember the second *Scientific Programming* class)
#  * Tokenize and lemmatize text

# The story only contains 5 lines. But each line can contain conversations, which are concatenated together in a single line.
# 
# ##### Note: If somebody wants to work with a file that contains more lines, you can use the file called `little-red-riding-hood-clean.txt`, which has more lines (conversations were not concatenated together).

# #### 2nd step:
# 
# Assuming you opened the file with 5 lines, for each paragraph you have to do the following:
# 
#  * Get synsets for all words (in English)
#  
#  * For each word, generate lemmas in a target language (and store them)
#  
#  * Choose 5 random words (make sure they have a target lemma)
#  
#  * For each of those random words, ask something that looks like this:
#  
#     * How youd you say the word `RANDOM_WORD` in Bulgarian (I use bulgarian as example, this can be any language of your choice)?
#     * Propose, then one correct lemma (from Wordnet) and other 4 random words (it doesn't matter where you get this random words, but they should be different in each test)
#     
# When you make your experiments, please tell me the target language you are using, so that I test it with that language.

# In[28]:

#Step 1

import re
def removePunctuation (word):
    return re.sub ("[^a-zA-Z0-9\s\-\']", "", word)

f = open("little-red-riding-hood-clean-5lines.txt", encoding="utf8")
lil_red=[removePunctuation(line.strip().lower()) for line in f] 
f.close()

l_red = ""
for paragraph in lil_red:
    l_red += paragraph + " "

tokens1 = nltk.word_tokenize(l_red)

from nltk.corpus import wordnet as wn

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN


tagged = nltk.pos_tag(tokens1)

wnl = nltk.WordNetLemmatizer()

lemmatized = []
for word in tagged:
    lemmatized.append(wnl.lemmatize(word[0], get_wordnet_pos(word[1])))

print(lemmatized)


# In[30]:

#Step 2 

#Get synsets for all words (English) For each word, generate lemmas in a target language (Spanish)

from nltk.corpus import wordnet as wn
import random as re

class wah:
    def __init__(self, syn_sets, lemmas):
        self.syn_sets = syn_sets
        self.lemmas = lemmas
        
    def __str__(self):
        return "%s --> %s" % (self.syn_sets, self.lemmas)
   
    def __repr__(self):
        return self.__str__()

    def hastranslation(self):
        if (len(self.lemmas) < 1):
            return False
        else:
            for lemma in self.lemmas:
                if (len(lemma) > 0):
                    return True
            return False
    
    def check(self, word):
        for lem in self.lemmas:
            for l in lem:
                if l == word:
                    return True
        return False
    
    def firsttranslation(self):
        if (self.hastranslation() == False):
            return None
    
        for lemma in self.lemmas:
            if (len(lemma) > 0):
                return lemma[0]
        
newdict = {}
for tok in tokens1:
    print(tok)
    syn_sets = wn.synsets(tok, lang='eng')
    lemmas = []
    for syn_set in syn_sets:
        lemmas.append(syn_set.lemma_names(lang='spa'))
    newdict[tok] = wah(syn_sets, lemmas)
    print(newdict[tok])


# In[31]:

#Step 2 

#Generate 5 random words and target language is SPANISH (SPA)
count = 0
rando_words = []
while count < 5:
    randindex = re.randint(0,len(newdict)-1)
    if list(newdict.values())[randindex].hastranslation() == False:
        continue
    count = count+1
    rando_words.append(list(newdict.keys())[randindex])

print(rando_words)


# In[32]:

#Step 2 Translate words to SPANISH

import sys 
for rando in rando_words:
    print("How do you say " + rando.upper() + " in spanish? ")
    correct_choice = newdict[rando].firsttranslation()
    
    count = 0
    mult_words = []
    while count < 4:
        randindex = re.randint(0,len(newdict)-1)
        if list(newdict.values())[randindex].hastranslation() == False:
            continue
        count = count+1
        mult_words.append(list(newdict.values())[randindex].firsttranslation())

        
    randominsertindex = re.randint(0, len(mult_words)-1)    
    mult_words.insert(randominsertindex, correct_choice)
    print(mult_words)
    
    choice = input("Your answer: ")
    if correct_choice == choice:
        print("Yay!\n")
    else:
        print("Thats wrong!\n")


# In[ ]:



