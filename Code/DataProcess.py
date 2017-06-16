#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:25:43 2017

@author: red-sky
"""

import sys
import json
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def process_ques(line, vocab):
    ques, answ, fact_num = line.split("\t")
    answ = answ.split(" ")
    ques_out = []
    num = 0
    for w in ques.replace("? ", "").split(" "):
        if not is_number(w):
            vocab.add(w)
            ques_out.append(w)
        else:
            num = int(w)
    return(num, (ques_out, answ), vocab)


def process_fact(line, vocab):
    fact = []
    num = 0
    for w in line.replace(".", "").split(" "):
        if not is_number(w):
            vocab.add(w)
            fact.append(w)
        else:
            num = int(w)
    return(num, fact, vocab)


def extractFromRaw(raw_txt):
    vocab = set()
    old_num = 0
    list_fact = []
    ques = []
    answ = []
    story = []
    for line in raw_txt:
        if "?" in line:
            new_num, processed_line, vocab = process_ques(line, vocab)
            q, a = processed_line[:]
            ques.append(q)
            answ.append(a)
            list_fact.append(story[:])
        else:
            new_num, processed_line, vocab = process_fact(line, vocab)
            if new_num >= old_num:
                story.append(processed_line)
            else:
                story = []
                story.append(processed_line)
        old_num = new_num
    vocab = ["PADDING"] + sorted(list(vocab))
    vocab_to_index = dict(zip(vocab, range(len(vocab))))
    index_to_vocab = dict(zip(range(len(vocab)), vocab))
    return(list_fact, ques, answ, [vocab_to_index, index_to_vocab])


def convert_list_words(list_words, vocab_to_index):
    re = [vocab_to_index[w] for w in list_words]
    return(re)


def convert_to_index(list_fact, ques, answ, vocab_to_index):
    indexed_facts, indexed_ques, indexed_answ = [], [], []

    for sample_facts, sample_ques, sample_answ in zip(list_fact, ques, answ):
        sample_facts = [
            convert_list_words(fact, vocab_to_index)
            for fact in sample_facts
        ]
        sample_ques = convert_list_words(sample_ques, vocab_to_index)
        sample_answ = convert_list_words(sample_answ, vocab_to_index)
        indexed_facts.append(sample_facts)
        indexed_ques.append(sample_ques)
        indexed_answ.append(sample_answ)
    return(indexed_facts, indexed_ques, indexed_answ)

if __name__ == "__main__":
    raw_txt = open(sys.argv[1], "r").read().splitlines()
    list_fact, ques, answ, vocab_map = extractFromRaw(raw_txt)
    indexed_data = convert_to_index(list_fact, ques, answ, vocab_map[0])
    with open(sys.argv[2], "w") as W:
        json.dump(vocab_map, W, indent=2)
    np.save(arr=indexed_data, file=sys.argv[3])
