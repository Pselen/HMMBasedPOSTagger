# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 00:38:55 2018

@author: Bilal
laplace
bigram
surface form
unk = 1 
"""
#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#%% 
# Initializatins
sentences = []
#%% 
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
# Extract sentences
sentence = [('<s>','<s>')]
for line in dataset:
    line_splited = line.split('\t')
    
    if len(line_splited) == 10:
        word = line_splited[1]
        pos = line_splited[3]
        # well, idk what are these underscores
        if word == '_' :
            continue
        sentence.append((word,pos))
    else:
        # end of sentence
        sentence.append(('</s>','</s>'))
        sentences.append(sentence)
        sentence = [('<s>','<s>')]
#%%
# viterbi 
def viterbi(input_sentence, transition_probabilities, observation_likelihoods, number_of_pos, word_dictionary, pos_dictionary):
    # viterbi    
    input_length = len(input_sentence) + 2
    number_of_states  = number_of_pos
    
    # intilize table, all zero 0
    viterbi_table = [[[0.0,0] for column in range(input_length)] for row in range(number_of_states)]
    viterbi_table[pos_dictionary['<s>']][0][0] = 1
    
    # states without start and end
    states = {k:v for k,v in pos_dictionary.items() if k != '<s>' and k != '</s>'}
    
    # fill the first column
    for state,state_idx in states.items():
        word_idx = word_dictionary.get(input_sentence[0], -1)
        start_state_idx = pos_dictionary['<s>']
        
        transition_p = transition_probabilities[start_state_idx][state_idx]
        # CHECK UNKOWN WORD
        # if word is not in the dictionary just ignore it by assigning it to 1
        obsservation_p = 1
        if word_idx != -1:
            obsservation_p = observation_likelihoods[state_idx][word_idx]
        
        viterbi_table[state_idx][1][0] = transition_p * obsservation_p
        viterbi_table[state_idx][1][1] = start_state_idx
    
    # fill all table, i=pre_state , j=state. zip for observation idx
    for word,v_t in zip(input_sentence[1:], range(2,len(input_sentence) + 1)):
        word_idx = word_dictionary.get(word, -1)
        for state,state_idx in states.items():
            list_to_get_max = []
            for pre_state,pre_state_idx in states.items():
                # p(state | pre_state) a_ij
                prob = transition_probabilities[pre_state_idx][state_idx]
                # v_t-1(i)
                v = viterbi_table[pre_state_idx][v_t-1][0]
                # v_t-1(i) * a_ij
                list_to_get_max.append((prob * v, pre_state_idx))
                
            # max( v_t-1(i) * a_ij) for i to N
            value, pointer = max(list_to_get_max)
            
            # CHECK UNKOWN WORD
            obsservation_p = 1
            if word_idx != -1:
                obsservation_p = observation_likelihoods[state_idx][word_idx]
            # v_t = max( v_t-1(i) * a_ij) for i to N * b_j(o_t)
            viterbi_table[state_idx][v_t][0] = value * obsservation_p
            viterbi_table[state_idx][v_t][1] = pointer
            
    # fill the last column, assume observation p( </s> | </s> ) = 1
    list_to_get_max = []
    end_state_idx = pos_dictionary['</s>']
    for pre_state,pre_state_idx in states.items():
        # p(state | pre_state) a_ij
        prob = transition_probabilities[pre_state_idx][end_state_idx]
        # v_t-1(i)
        v = viterbi_table[pre_state_idx][-2][0]
        # v_t-1(i) * a_ij
        list_to_get_max.append((prob * v, pre_state_idx))
    # max( v_t-1(i) * a_ij) for i to N
    value, pointer = max(list_to_get_max)
    # v_t = max( v_t-1(i) * a_ij) for i to N * b_j(o_t)
    viterbi_table[end_state_idx][-1][0] = value 
    viterbi_table[end_state_idx][-1][1] = pointer
          
    
    # backtrack, extract path
    pointer = viterbi_table[end_state_idx][-1][1] 
    state_order = [pointer]
    for i in reversed(range(2,input_length - 1)):
        pointer = viterbi_table[pointer][i][1]
        state_order.append(pointer)
    pos_dictionary_rev = {v: k for k, v in pos_dictionary.items()}
    output = [pos_dictionary_rev[i] for i in reversed(state_order)]
    return output        
#%%
accuracies = []

kf = KFold(n_splits=10,random_state=1, shuffle=False)
for index_train, index_test in kf.split(sentences):
    
    sentences_train = [sentences[i] for i in index_train]
    sentences_test = [sentences[i] for i in index_test]
    
    word_dictionary = {}
    pos_dictionary = {}
    
    observation_counts = np.empty((0))
    transition_counts = np.empty((0))
    
    observation_likelihoods = np.empty((0))
    transition_probabilities = np.empty((0))
    
    # extract all unique word&pos
    for sentence in sentences_train:
        for word,pos in sentence:
            if word != '<s>' and word != '</s>':
                word_dictionary[word] = 0
                pos_dictionary[pos] = 0
    
    # last two index
    pos_dictionary['<s>'] = 0
    pos_dictionary['</s>'] = 0       
    
    number_of_words = len(word_dictionary.keys())
    number_of_pos = len(pos_dictionary.keys())
    
    # assign them an unique index    
    for key,index in zip(word_dictionary.keys(), range(number_of_words)):
        word_dictionary[key] = index
    
    for key,index in zip(pos_dictionary.keys(), range(number_of_pos)):
        pos_dictionary[key] = index
    
    # count observations
    observation_counts = np.zeros((number_of_pos-2, number_of_words), dtype=np.float32)
    for sentence in sentences_train:
        for word,pos in sentence:
                if word != '<s>' and word != '</s>':
                    observation_counts[pos_dictionary[pos]][word_dictionary[word]] += 1
       
    # count transitions
    transition_counts = np.zeros((number_of_pos, number_of_pos), dtype=np.float32)
    for sentence in sentences_train:
        for pos_0,pos_1 in zip(sentence,sentence[1:]):
            transition_counts[pos_dictionary[pos_0[1]]][pos_dictionary[pos_1[1]]] += 1
    
    # apply smoothing, be carefull with </s> 15th row(left as zero) and <s> 14th column(left as zero)
    # simple laplace smoothing
    # TODO: OTHER SMOOTHS
    transition_counts = transition_counts + 1 
    transition_counts[:,14] = 0
    transition_counts[15,:] = 0    
    
    # calculate probabilities
    # implementation detail(create seperate dictinary for each dimension,
    # makes more sense if elements got different values in each dim)
    transition_probabilities = transition_counts / np.sum(transition_counts, axis=1).reshape(-1,1)
    transition_probabilities[np.isnan(transition_probabilities)] = 0
    observation_likelihoods = observation_counts / np.sum(observation_counts, axis=1).reshape(-1,1)
    
    correctly_tagged_word = 0.0
    incorrectly_tagged_word = 0.0
    correctly_tagged_sentence = 0.0
    incorrectly_tagged_sentence = 0.0
    
    for sentence in sentences_test:
        input_sentence = [i[0] for i in sentence[1:-1]] 
        target_tags = [i[1] for i in sentence[1:-1]] 
        
        output = viterbi(input_sentence=input_sentence, transition_probabilities=transition_probabilities,
                         observation_likelihoods=observation_likelihoods,
                         number_of_pos=number_of_pos, word_dictionary=word_dictionary, pos_dictionary=pos_dictionary)
        all_correct = True
        for i,j in zip(output,target_tags):
            if i==j:
                correctly_tagged_word += 1
            else:
                incorrectly_tagged_word += 1
                all_correct = False
        
        if all_correct:
            correctly_tagged_sentence += 1
        else:
            incorrectly_tagged_sentence += 1
        
    accuracy_words = correctly_tagged_word / (correctly_tagged_word + incorrectly_tagged_word)
    accuracy_sentences = correctly_tagged_sentence / (correctly_tagged_sentence + incorrectly_tagged_sentence)
    
    accuracies.append((accuracy_words,accuracy_sentences))
    
    
print(accuracies)
print(sum([i[0] for i in accuracies]) / 10.0)
print(sum([i[1] for i in accuracies]) / 10.0)
