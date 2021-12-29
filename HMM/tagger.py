import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    # for data in train_data:
    #     print(data.words)
    #     print(data.tags)
    for idx,tag in enumerate(tags):
        tag2idx[tag] = idx
    for idx,word in enumerate(unique_words):
        word2idx[word] = idx
    print(word2idx)
    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    
    for data in train_data:
        pi[tag2idx[data.tags[0]]] +=1
        # print(f"words {len(data.words)}")
        # print(f"tags {len(data.tags)}")
        B[tag2idx[data.tags[0]],word2idx[data.words[0]]] +=1
        for n in range(1,len(data.words)):
            A[tag2idx[data.tags[n-1]],tag2idx[data.tags[n]]] +=1
            B[tag2idx[data.tags[n]],word2idx[data.words[n]]] +=1

    pi = pi/np.sum(pi)
    A = A/np.sum(A,axis=1).reshape(-1,1)
    B = B/np.sum(B,axis=1).reshape(-1,1)
    
    # print(A)
    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    S = len(model.pi)
    for data in test_data:
        for word_no in range(len(data.words)):
            if data.words[word_no] not in model.obs_dict:
                tag_idx = model.state_dict[data.tags[word_no]]
                model.obs_dict[data.words[word_no]] = len(model.obs_dict)
                # word_idx = model.obs_dict[data.words[word_no]]
                column = np.ones((S,1))*1e-6
                model.B = np.append(model.B,column,axis=1)
        
        tagging.append(model.viterbi(data.words))

    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
