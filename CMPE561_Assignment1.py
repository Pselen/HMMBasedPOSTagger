# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 00:38:55 2018

@author: Selen & Bilal
laplace
bigram
stemmed forms
unk = 1 
"""
#%%
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
#%% 
# Read input file as dataset and return the sentences
def read_file(filename, original_or_stem):
    with open(filename, 'r', encoding='utf-8') as f:
        dataset = f.readlines()

    # Correct erronous tags of word satın
    for d in dataset:
        if len(d.split('\t'))>1:
            word = d.split('\t')[original_or_stem]
            if word == 'satın' or word == 'Satın':
                tag = 'Noun'
                new_d = d.split('\t')[0] + '\t' + d.split('\t')[1] + '\t' + d.split('\t')[2]  + '\t' + tag + '\t' + tag + '\t' + d.split('\t')[5]+ '\t' + d.split('\t')[6] + '\t' + d.split('\t')[7]  + '\t' + d.split('\t')[8] + '\t' + d.split('\t')[9] 
                dataset[dataset.index(d)] = new_d
    # Extract sentences from dataset
    sentence = [('<s>','<s>','<s>')]
    for line in dataset:
        line_splited = line.split('\t')
        
        if len(line_splited) == 10:
            word = line_splited[original_or_stem]
            pos = line_splited[3]
            suffix = line_splited[5]
            # Skip the underscores
            if word == '_' :
                continue
            sentence.append((word,pos,suffix))
        else:
            # End of sentence
            sentence.append(('</s>','</s>','</s>'))
            sentences.append(sentence)
            sentence = [('<s>','<s>','<s>')]
    return sentences
#%%
## Viterbi Algorithm
def viterbi(input_sentence, transition_probabilities, observation_likelihoods, number_of_pos, word_dictionary, pos_dictionary,
            morph_probabilities,morph_dictionary,morph_inf):
    
    input_morphs = [ i[1] for i in input_sentence] 
    input_sentence = [ i[0] for i in input_sentence] 
    input_length = len(input_sentence) + 2
    number_of_states  = number_of_pos
    
    # Intialize table, all zero 0
    viterbi_table = [[[0.0,0] for column in range(input_length)] for row in range(number_of_states)]
    viterbi_table[pos_dictionary['<s>']][0][0] = 1
    
    # States without start and end
    states = {k:v for k,v in pos_dictionary.items() if k != '<s>' and k != '</s>'}
    
    # Fill in the first column
    for state,state_idx in states.items():
        word_idx = word_dictionary.get(input_sentence[0], -1)
        start_state_idx = pos_dictionary['<s>']
        
        transition_p = transition_probabilities[start_state_idx][state_idx]
        # CHECK UNKOWN WORD
        # If word is not in the dictionary just ignore it by assigning it to 1
        obsservation_p = 1
        if word_idx != -1:
            obsservation_p = observation_likelihoods[state_idx][word_idx]
        else:
            if morph_inf:
                if len(input_morphs[0].split('|')) > 0:
                    tot = 1
                    for j in input_morphs[0].split('|'):
                        morf_idx = morph_dictionary.get(j, -1)
                        if morf_idx != -1:  
                            tot *= morph_probabilities[state_idx][morf_idx]
                    obsservation_p = tot
        
        viterbi_table[state_idx][1][0] = transition_p * obsservation_p
        viterbi_table[state_idx][1][1] = start_state_idx
    
    # Fill all table, i=pre_state , j=state. zip for observation idx
    for word,v_t,morf in zip(input_sentence[1:], range(2,len(input_sentence) + 1),input_morphs[1:]):
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
            else:
                if morph_inf:
                    if len(morf.split('|')) > 0:
                        tot = 1
                        for j in morf.split('|'):
                            morf_idx = morph_dictionary.get(j, -1)
                            if morf_idx != -1:  
                                tot *= morph_probabilities[state_idx][morf_idx]
                        obsservation_p = tot
            # v_t = max( v_t-1(i) * a_ij) for i to N * b_j(o_t)
            viterbi_table[state_idx][v_t][0] = value * obsservation_p
            viterbi_table[state_idx][v_t][1] = pointer
            
    # Fill in the last column, assume observation p( </s> | </s> ) = 1
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
          
    # Backtrack and extract the path
    pointer = viterbi_table[end_state_idx][-1][1] 
    state_order = [pointer]
    for i in reversed(range(2,input_length - 1)):
        pointer = viterbi_table[pointer][i][1]
        state_order.append(pointer)
    pos_dictionary_rev = {v: k for k, v in pos_dictionary.items()}
    output = [pos_dictionary_rev[i] for i in reversed(state_order)]
    # Return the path which is the most probable one
    return output        
#%%
def evaluation(sentences,original_or_stem,morph_inf):
    if original_or_stem == 1:
        end_position = 15
    else:
        end_position = 14
    accuracy_of_tagging_words = 0.0
    accuracy_of_tagging_sentences = 0.0
    number_of_folds = 10
    kf = KFold(n_splits=number_of_folds,random_state=1, shuffle=False)
    cumulative_confusion_matrix = np.zeros(15)
    
    for index_train, index_test in kf.split(sentences):
        word_dictionary = {}
        pos_dictionary = {}
        morph_dictionary = {}
        
        observation_counts = np.empty((0))
        transition_counts = np.empty((0))
        
        observation_likelihoods = np.empty((0))
        transition_probabilities = np.empty((0))
        
        sentences_train = [sentences[i] for i in index_train]
        sentences_test = [sentences[i] for i in index_test]
        
        words = []
        # extract all unique word&pos&morphs
        for sentence in sentences_train:
            for word,pos,morph in sentence:
                if word != '<s>' and word != '</s>':
                    word_dictionary[word] = 0
                    pos_dictionary[pos] = 0
                    words.append((word,pos,morph))
                    if morph != '_':
                        for j in morph.split('|'):
                            morph_dictionary[j] = 0
        
        # last two index
        # We should ignore </s> column (it may worsen the prediction though)
        pos_dictionary['<s>'] = 0
        pos_dictionary['</s>'] = 0       
        
        number_of_words = len(word_dictionary.keys())
        number_of_pos = len(pos_dictionary.keys())
        
        # Assign them a unique index    
        for key,index in zip(word_dictionary.keys(), range(number_of_words)):
            word_dictionary[key] = index
        
        for key,index in zip(pos_dictionary.keys(), range(number_of_pos)):
            pos_dictionary[key] = index
        
        for key,index in zip(morph_dictionary.keys(), range(len(morph_dictionary.keys()))):
            morph_dictionary[key] = index
        
        # Count observations
        observation_counts = np.zeros((number_of_pos-2, number_of_words), dtype=np.float32)
        for sentence in sentences_train:
            for word,pos,_  in sentence:
                    if word != '<s>' and word != '</s>':
                        observation_counts[pos_dictionary[pos]][word_dictionary[word]] += 1
           
        # Count transitions
        transition_counts = np.zeros((number_of_pos, number_of_pos), dtype=np.float32)
        for sentence in sentences_train:
            for pos_0,pos_1 in zip(sentence,sentence[1:]):
                transition_counts[pos_dictionary[pos_0[1]]][pos_dictionary[pos_1[1]]] += 1
                
        # Count morphs
        morph_counts = np.zeros((number_of_pos-2, len(morph_dictionary.keys())), dtype=np.float32)
        for word,pos,m in words:
            for j in  m.split('|'):
                if j != '_':
                    morph_counts[pos_dictionary[pos]][morph_dictionary[j]] += 1
        
        # Apply smoothing, be carefull with </s> 15th row(left as zero) and <s> 14th column(left as zero)
        # laplace smoothing
        transition_counts = transition_counts + 1 
        transition_counts[:,end_position-1] = 0
        transition_counts[end_position,:] = 0
        # morpfs_counts = morph_counts + 1   
        
        # Calculate probabilities
        # implementation detail(create seperate dictinary for each dimension,
        # makes more sense if elements got different values in each dim)
        transition_probabilities = transition_counts / np.sum(transition_counts, axis=1).reshape(-1,1)
        transition_probabilities[np.isnan(transition_probabilities)] = 0
        observation_likelihoods = observation_counts / np.sum(observation_counts, axis=1).reshape(-1,1)
        morph_probabilities = morph_counts / np.sum(morph_counts, axis=1).reshape(-1,1)
        morph_probabilities[np.isnan(morph_probabilities)] = 0
            
        correctly_tagged_word = 0.0
        incorrectly_tagged_word = 0.0
        correctly_tagged_sentence = 0.0
        incorrectly_tagged_sentence = 0.0
        list_target_tags, list_output = [], []
        
        for sentence in sentences_test:
            input_sentence = [(i[0],i[2]) for i in sentence[1:-1]] 
            target_tags = [i[1] for i in sentence[1:-1]] 
            list_target_tags += target_tags
            output = viterbi(input_sentence=input_sentence, transition_probabilities=transition_probabilities,
                         observation_likelihoods=observation_likelihoods,
                         number_of_pos=number_of_pos, word_dictionary=word_dictionary, pos_dictionary=pos_dictionary,
                          morph_probabilities=morph_probabilities,morph_dictionary=morph_dictionary,morph_inf=morph_inf)
            list_output += output
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
                
        accuracy_of_tagging_words += correctly_tagged_word / (correctly_tagged_word + incorrectly_tagged_word)
        accuracy_of_tagging_sentences += correctly_tagged_sentence / (correctly_tagged_sentence + incorrectly_tagged_sentence)
        
        cf = confusion_matrix(list_target_tags, list_output, labels=['</s>','<s>','Adj','Adv','Conj','Det','Dup','Interj','Noun','Num','Postp','Pron','Punc','Ques','Verb'])
        cumulative_confusion_matrix = np.add(cumulative_confusion_matrix,cf)
        
    final_confusion_matrix = np.true_divide(cumulative_confusion_matrix,float(number_of_folds))
    final_accuracy_of_tagging_words = accuracy_of_tagging_words/number_of_folds
    final_accuracy_of_tagging_sentences= accuracy_of_tagging_sentences/number_of_folds
    
    return(final_accuracy_of_tagging_words, final_accuracy_of_tagging_sentences, final_confusion_matrix )
#%% 
sentences = []
# To use original word form = 1, to use lemma form =2.
original_or_stem = 1
morph_inf = True
filename = r"Project (Application 1) (MetuSabanci Treebank).conll"

sentences = read_file(filename, original_or_stem)
accuracy_of_tagging_words, accuracy_of_tagging_sentences, final_confusion_matrix = evaluation(sentences, original_or_stem, morph_inf)
