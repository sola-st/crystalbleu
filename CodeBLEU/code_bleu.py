# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
# import argparse
from CodeBLEU import bleu
from CodeBLEU import weighted_ngram_match
from CodeBLEU import syntax_match
from CodeBLEU import dataflow_match
import json

def make_weights(reference_tokens, key_word_list):
    return {token:1 if token in key_word_list else 0.2 \
            for token in reference_tokens}

def code_bleu(refs, hyp, lang='java', params='0.25,0.25,0.25,0.25'):
    alpha,beta,gamma,theta = [float(x) for x in params.split(',')]
    sm_func = bleu.SmoothingFunction(epsilon=0.0001).method1
    tokenized_hyps = hyp
    tokenized_refs = refs
    hypothesis = []
    references = []
    for i in tokenized_hyps:
        hypothesis.append(' '.join(i))
    for i in tokenized_refs:
        references.append([' '.join(i[0])])

    ngram_match_score = bleu.corpus_bleu(tokenized_refs,tokenized_hyps, smoothing_function=sm_func)

    # calculate weighted ngram match
    keywords = [x.strip() for x in open('CodeBLEU/keywords/'+lang+'.txt', 'r', encoding='utf-8').readlines()]
    
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    # print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.\
    #                     format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

    code_bleu_score = alpha*ngram_match_score\
                    + beta*weighted_ngram_match_score\
                    + gamma*syntax_match_score\
                    + theta*dataflow_match_score

    # print('CodeBLEU score: ', code_bleu_score)
    return code_bleu_score




