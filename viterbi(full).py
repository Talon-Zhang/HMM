# viterbi.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Renxuan Wang (renxuan2@illinois.edu) on 10/18/2018

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

'''
TODO: implement the baseline algorithm.
input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
output: list of sentences, each sentence is a list of (word,tag) pairs. 
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
from math import *
import numpy as np

def baseline(train, test):
    predicts = []

    # store the probability P(word|tag), key is the tuple (word,tag). 
    wt_dict={}
    # store the total number of words in for each tag (may be repeated words)
    wtot_dict={}

    tags=[]

    for sentence in train:
        for i in range(len(sentence)):
            pair=sentence[i]

            if pair[-1] not in tags:
                tags.append(pair[-1])

            if pair in wt_dict:
                wt_dict[pair]+=1
            else:
                wt_dict[pair]=1

            if pair[-1] in wtot_dict:
                wtot_dict[pair[-1]]+=1
            else:
                wtot_dict[pair[-1]]=1

    for key in wt_dict.keys():
        tag=key[-1]
        n=wt_dict[key]
        wt_dict[key]=log10(n/wtot_dict[tag])

    most_freq_tag='hhh'
    most_freq=-1
    for tag in tags:
        if wtot_dict[tag]>most_freq:
            most_freq=wtot_dict[tag]
            most_freq_tag=tag


    def helper(sentence):
        if len(sentence)==0:
            return[]
        
        rlist=[]
        for word in sentence:
            max_prob=-10000
            max_pair=(word,0)
            for tag in tags:
                pair=(word, tag)
                if pair in wt_dict:
                    if wt_dict[pair]>max_prob:
                        max_prob=wt_dict[pair]
                        max_pair=pair
            if max_pair[-1]==0:
                max_pair=(word, most_freq_tag)

            rlist.append(max_pair)

        return rlist


    for s in test:
        predicts.append(helper(s))
    

    return predicts

'''
TODO: implement the Viterbi algorithm.
input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
output: list of sentences with tags on the words
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
def viterbi(train, test):
    
    predicts = []

    #start_time = time.time()
    # smoothing parameter for the unseen word (emission probability). Use laplace smoothing here
    smoothing_e=0.00001
    # smoothing parameter for the unseen tag pair (transition probability). Use laplace smoothing here
    smoothing_t=0.00005
    # smoothing parameter for the tag if it has not been seen to appear at first. Use laplace smoothing
    smoothing_i=0.0001
    # store all the possible tags appeared in the training data
    tags=[]
    # store the probability that a tag appears at the start of the sentence
    p_i_dict={}
    # store the probability P(word|tag), key is the tuple (word,tag). Emission probability
    wt_dict={}
    # store the probability P(t_k|t_k-1), key is the tuple (t_k, t_k-1). Transition probability
    tt_dict={}
    # store the total number of different words in each tag. Need this to normalize the emission probability after smoothing
    tnum_dict={}
    # store the total number of different t_k for each t_k-1. Need this to normalize the transition probability after smoothing
    tagnum_dict={}
    # store the total number of words in for each tag (may be repeated words)
    wtot_dict={}
    # store the total number of t_k for each t_k-1
    ttot_dict={}


    for sentence in train:
        for i in range(len(sentence)):
            pair=sentence[i]
            
            if pair[-1] not in tags:
                tags.append(pair[-1])

            if i==0: 
            # it's at the beginning of the sentence
                if pair[-1] in p_i_dict:
                    p_i_dict[pair[-1]]+=1
                else:
                    p_i_dict[pair[-1]]=1+smoothing_i

            if pair in wt_dict:
                wt_dict[pair]+=1
            else:
                wt_dict[pair]=1+smoothing_e
                if pair[-1] in tnum_dict:
                    tnum_dict[pair[-1]]+=1
                else:
                    tnum_dict[pair[-1]]=1

            if pair[-1] in wtot_dict:
                wtot_dict[pair[-1]]+=1
            else:
                wtot_dict[pair[-1]]=1


            if i!=0:
                tag_pair=(sentence[i][-1], sentence[i-1][-1])
                if tag_pair in tt_dict:
                    tt_dict[tag_pair]+=1
                else:
                    tt_dict[tag_pair]=1+smoothing_t
                    if tag_pair[-1] in tagnum_dict:
                        tagnum_dict[tag_pair[-1]]+=1
                    else:
                        tagnum_dict[tag_pair[-1]]=1

                if tag_pair[-1] in ttot_dict:
                    ttot_dict[tag_pair[-1]]+=1
                else:
                    ttot_dict[tag_pair[-1]]=1


    demoninator_initial=len(train)+smoothing_i*(len(p_i_dict)+1)
    for key in p_i_dict.keys():
        n=p_i_dict[key]
        p_i_dict[key]=log10(n/demoninator_initial)
    p_i_dict['UNK']=log10(smoothing_i/demoninator_initial)

    denominator_emission={}
    for tag in wtot_dict.keys():
        denominator_emission[tag]=wtot_dict[tag]+smoothing_e*(tnum_dict[tag]+1)
    for key in wt_dict.keys():
        tag=key[-1]
        n=wt_dict[key]
        wt_dict[key]=log10(n/denominator_emission[tag])
    for tag in denominator_emission.keys():
        wt_dict[('UNK', tag)]=log10(smoothing_e/denominator_emission[tag])


    denominator_transition={}
    for tag in ttot_dict.keys():
        denominator_transition[tag]=ttot_dict[tag]+smoothing_t*(tagnum_dict[tag]+1)
    for key in tt_dict.keys():
        tag=key[-1]
        n=tt_dict[key]
        tt_dict[key]=log10(n/denominator_transition[tag])
    for tag in denominator_transition.keys():
        tt_dict[('UNK', tag)]=log10(smoothing_t/denominator_transition[tag])
        
    

    #end_time = time.time()
    #print(str(end_time-start_time)+' sec')
    
    # helper function. Building the trellis. 
    def helper(sentence):
        # the array of trellis. When completed, each array should store the tuple (highest probability, 
        # index of the previous tag that gives the highest probability)
        if len(sentence)==0:
            return[]

        trellis=np.zeros((len(tags),len(sentence)),dtype=[('f1',np.float32),('f2',np.int16)])

        for i in range(len(tags)):
            if tags[i] in p_i_dict:
                initial=p_i_dict[tags[i]]
            else:
                initial=p_i_dict['UNK']

            if (sentence[0], tags[i]) in wt_dict:
                trellis[i,0]=(wt_dict[(sentence[0], tags[i])]+initial,-20)
            else:
                trellis[i,0]=(wt_dict[('UNK', tags[i])]+initial,-20)

        for column in range(1, len(sentence)):
            for row in range(len(tags)):
                cur_tag=tags[row]

                if (sentence[column], tags[row]) in wt_dict:
                    emission=wt_dict[(sentence[column], tags[row])]
                else:
                    emission=wt_dict[('UNK', tags[row])]

                # find the largest transition probability+previous trellis cell
                max_transtion=(-1000, 'meow')
                for k in range(len(tags)):
                    pre_tag=tags[k]
                    if (cur_tag, pre_tag) in tt_dict:
                        transition=tt_dict[(cur_tag, pre_tag)]+trellis[k,column-1][0]
                    else:
                        transition=tt_dict[('UNK', pre_tag)]+trellis[k,column-1][0]

                    if transition>max_transtion[0]:
                        max_transtion=(transition, k)

                trellis[row, column]=(emission+max_transtion[0],max_transtion[-1])

        tag_list=[]
        n=len(sentence)-1
        index=np.argmax(np.array([i[0] for i in list(trellis[:,n])]))
        tag_list.append((sentence[n],tags[index]))
        new_index=trellis[index,n][-1]
        while len(tag_list)!=len(sentence):
            n-=1
            tag_list.append((sentence[n],tags[new_index]))
            new_index=trellis[new_index,n][-1]

        tag_list.reverse()

        return tag_list

    for s in test:
        #start_time1=time.time()
        predicts.append(helper(s))
        #end_time1 = time.time()
        #print(str(end_time1-start_time1)+' sec')



    #print(len(test))
    #end_time = time.time()
    #print(str(end_time-start_time)+' sec')
    return predicts
