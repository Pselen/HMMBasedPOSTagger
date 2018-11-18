"""
Created on Sun Nov 18 18:43:14 2018

@author: parla
"""
#%%
from math import log
#%%
observation = {} # Observation dictionary which holds (current_word, current_tag) pairs as keys and their number of occuraces as value.
transition = {} # Transition dictionary which holds (current_tag-previous_tag) pairs as keys and their number of occuraces as value.
tags = {} # Tags dictionary which holds (tag, count) pairs.
#encoding:utf-8
# -*- coding: utf-8 -*
# -*- coding: utf-8 -*-
with open(r"Project (Application 1) (MetuSabanci Treebank).conll", 'r', encoding='utf-8') as f:
    dataset = f.readlines()
#%% 
# Correct erronous tags of word satın
for d in dataset:
    if len(d.split('\t'))>1:
        word = d.split('\t')[1]
        if word == 'satın' or word == 'Satın':
            tag = 'Noun'
            new_d = d.split('\t')[0] + '\t' + d.split('\t')[1] + '\t' + d.split('\t')[2]  + '\t' + tag + '\t' + tag + '\t' + d.split('\t')[5]+ '\t' + d.split('\t')[6] + '\t' + d.split('\t')[7]  + '\t' + d.split('\t')[8] + '\t' + d.split('\t')[9] 
            dataset[dataset.index(d)] = new_d
#%% 
# Correct erronous underscores.
for d in dataset:
    if len(d.split('\t'))>1:
        if d.split('\t')[1] == '_':
            dataset.remove(dataset[dataset.index(d)])
        elif d.split('\t')[1] == '_' and d.split('\t')[2]  == '_':
            dataset.remove(dataset[dataset.index(d)])
#%%
current_tag, current_word = '', ''
number_of_sentences = 0 # number_of_sentences is same as the count of 'START' tag.
previous_tag = 'START' # Concatenate 'START' tag at the beginning of each sentence.
tags['END'] = 0 # Concatenate 'END' tag at the end of each sentence.
sentence = ''
sentences = []
tags_of_sentence = ''
tags_of_sentences = []

# TODO: Stores bigrams, add also trigrams.
# Parse dataset line by line.
for line in dataset:
    # Each word is seperated by '\t' in the document. 
    word = line.split('\t')
    # Loop till the end of a sentence
    if len(word)==10:
        # POS tag lies at the 4th index of current line.
        current_tag = word[3]
        # TODO: Convert each word tolower case letter.
        current_word = word[1]#.lower()
        # If current line is empty, skip
        if current_word != '_' :
            word_tag_pair = (current_word, current_tag)
        # If word tag pair is not listed in observation list, add it to list; otherwise, increment it.
        if word_tag_pair in observation:
            observation[word_tag_pair] = observation[word_tag_pair] + 1
        else:
            observation[word_tag_pair] = 1
        
        tag = current_tag
        # If tag is not listed in tags list, add it to list; otherwise, increment it.
        if tag in tags:
            tags[tag] = tags[tag] + 1
        else:
            tags[tag] = 1

        tag_tag_pair = (current_tag, previous_tag)
        # If tag tag pair is not listed in transition list, add it to list; otherwise, increment it.
        if tag_tag_pair in transition:
            transition[tag_tag_pair] = transition[tag_tag_pair] + 1
        else:
            transition[tag_tag_pair] = 1	
        # Iterate tag by one
        previous_tag = current_tag
        sentence += word[1] + ' '
        tags_of_sentence += tag + ' '        
    # At the end of the sentence, add an 'END' tag and assign 'START' to the previous_tag in order to skip count END-START probability.
    else:
        tags['END']+=1
        end_tag_pair = ('END', previous_tag)
        if end_tag_pair in transition:
            transition[end_tag_pair]+=1
        else:
            transition[end_tag_pair] = 1
	
        previous_tag = 'START'
        number_of_sentences+=1
        sentences.append(sentence)
        sentence = ''
        tags_of_sentences.append(tags_of_sentence)
        tags_of_sentence = ''
tags['START'] = number_of_sentences 
#%%
# Calculate observation probabilities
# no. of occurrences of a given word-tag divided by no. of occurrences of the tag.
observation_probabilities = {}
for (word_tag_pair,count_word_tag_pair) in observation.items():
    tag = word_tag_pair[1]
    observation_probabilities[word_tag_pair] = float(count_word_tag_pair)/float(tags[tag])
#%%
# Calculate transition probabilities 
# Adding one to numerator and unique tag counts to denominator for smoothing; Laplace
# TODO: Try other smoothing techniques.
# TODO: Tro for trigrams
transition_probabilities = {}
for (tag_tag_pair,count_tag_tag_pair) in transition.items():
    prev_tag = tag_tag_pair[0]
    transition_probabilities[tag_tag_pair] = float(count_tag_tag_pair+1)/float(tags[prev_tag] + len(tags))
#%%
#TODO: Implement Viterbi algortihm
