#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:01:13 2017

@author: red-sky
"""

import os
import sys
import json
import pickle
import numpy
import time
import theano
import theano.tensor as T

from PreInputModule import EncodingLayer
from BiDirectionGRU import InputModule
from EpisodicMemoriesModule import EpisodicModule
from utils import LogisticRegression, get_groups, get_batch, \
                    ADAM_OPTIMIZER, DropOut, padding


def main(data_train_path, data_validate_path, vocabulary_path,
         trainedParamsPath=None, mode="TRAIN", data_group_len=3,
         batchsize=128, n_epoch=512, learning_rate=0.001,
         l2_regu_rate=0.001, word_dim=80, hiden_dim=80, keep_rate=0.9):

    # CONSTANT
    RNG = numpy.random.RandomState(220495)
    N_NON_WORD = 1
    TRAINNING = 1.0
    VALIDATING = 0.0
    # collect data for training process
    word_to_index, index_to_word = json.load(open(vocabulary_path, "r"))
    vocab = numpy.array(list(word_to_index.keys()))
    facts, ques, answ = numpy.load(data_train_path)
    data_train = numpy.dstack((facts, ques, answ))[0]

    facts, ques, answ = numpy.load(data_validate_path)
    data_val = numpy.dstack((facts, ques, answ))[0]

    groups_data_train = get_groups(data_train, groups_len=data_group_len)
    groups_data_val = get_groups(data_val, groups_len=data_group_len)

    ####################
    # LOAD MODEL HERE  #
    ####################

    if os.path.isfile(trainedParamsPath):
        with open(trainedParamsPath, 'rb') as handle:
            trainedParams = pickle.load(handle)
    else:
        print("No Trained Model, create new")
        trainedParams = {
            'EMBD': None,
            'INPUT_FUSION_LAYER': None,
            'EPISODIC_MEM_PASS1': None,
            'EPISODIC_MEM_PASS2': None,
            'EPISODIC_MEM_PASS3': None,
            'prediction_layer': None
        }

    ####################
    # BUILD MODEL HERE #
    ####################

    # input tensor
    tensor_facts = T.itensor3()
    tensor_question = T.imatrix()
    train_state = T.dscalar()
    keep_rate = theano.shared(
        numpy.asarray(keep_rate, dtype=theano.config.floatX),
        borrow=True
    )
    # Words embedding and Encoding scheme for sentences (facts), question, v.v
    EMBD = EncodingLayer(num_vocab=len(vocab) - N_NON_WORD,
                         word_dim=word_dim, rng=RNG,
                         embedding_w=trainedParams["EMBD"])
    dropout_layer = DropOut(RNG=RNG)

    # positional encoding scheme for list of facts
    tensor_facts_embded = EMBD.sents_ind_2vec(tensor_facts)
    tensor_question_embded = EMBD.words_ind_2vec(tensor_question)

    tensor_facts_embded = dropout_layer.output(
        layer_input=tensor_facts_embded,
        keep_rate=keep_rate,
        train=train_state
    )

    # INPUT MODULE --- Input Fusion Layer + Bi-Directional GRU
    INPUT_FUSION_LAYER = InputModule(
        RNG_=RNG, paramsTrained=trainedParams["INPUT_FUSION_LAYER"],
        input_sents=tensor_facts_embded,
        quesion=tensor_question_embded,
        num_in=word_dim, num_out=hiden_dim,
        name="INPUT_FUSION_LAYER"
    )

    # INPUT_FUSION_LAYER output -- Bi-directional GRU facts and GRU of question
    bi_directional_facts, gru_question = INPUT_FUSION_LAYER.output

    # EPISODIC MEMORY NETWORK LAYERS --- Attention gate in GRU plus
    #                                         + Memory Update
    EPISODIC_MEM_PASS1 = EpisodicModule(
        RNG=RNG,
        fact_dim=hiden_dim, context_dim=160, mem_dim=hiden_dim,
        paramsTrained=trainedParams["EPISODIC_MEM_PASS1"],
        name="EPISODIC_MEM_PASS1_"
    )
    EPISODIC_MEM_PASS2 = EpisodicModule(
        RNG=RNG,
        fact_dim=hiden_dim, context_dim=160, mem_dim=hiden_dim,
        paramsTrained=trainedParams["EPISODIC_MEM_PASS2"],
        name="EPISODIC_MEM_PASS2_"
    )
    EPISODIC_MEM_PASS3 = EpisodicModule(
        RNG=RNG,
        fact_dim=hiden_dim, context_dim=160, mem_dim=hiden_dim,
        paramsTrained=trainedParams["EPISODIC_MEM_PASS3"],
        name="EPISODIC_MEM_PASS3_"
    )

    # create initial memmory vector
    begin_memory_state = T.identity_like(gru_question)
    begin_memory_state = T.fill(begin_memory_state, gru_question)

    # PASS 1
    episodic_mem_pass1, atten1 = EPISODIC_MEM_PASS1.output(
        bi_directional_facts, gru_question, begin_memory_state
    )

    # PASS 1
    episodic_mem_pass2, atten2 = EPISODIC_MEM_PASS2.output(
        bi_directional_facts, gru_question, episodic_mem_pass1
    )

    # PASS 1
    episodic_mem_pass3, atten3 = EPISODIC_MEM_PASS3.output(
        bi_directional_facts, gru_question, episodic_mem_pass2
    )

    # dropout_layer for answer module
    episodic_mem_pass3 = dropout_layer.output(
        layer_input=episodic_mem_pass3,
        keep_rate=keep_rate,
        train=train_state
    )

    # Prediction layer LogisticRegression
    prediction_layer = LogisticRegression(
        rng=RNG, paramsLayer=trainedParams["prediction_layer"],
        layerInput=episodic_mem_pass3,
        n_in=hiden_dim, n_out=len(vocab) - N_NON_WORD,
        name="prediction_layer_"
    )

    if (mode == "TRAIN"):
        tensor_answers = T.imatrix()
        # COST FUNCTION --- negative log likelihood function
        n_sample = tensor_answers.shape[0]
        true_label = (tensor_answers - N_NON_WORD).reshape(shape=(n_sample, ))

        # l2 regulation
        L2 = (INPUT_FUSION_LAYER.L2 +
              EPISODIC_MEM_PASS1.L2 +
              EPISODIC_MEM_PASS2.L2 +
              EPISODIC_MEM_PASS3.L2 +
              prediction_layer.L2)

        # Negative log likelihood
        NLL_loss = prediction_layer.neg_log_likelihood(true_label)

        # COST FUNTION
        COST_VALUE = NLL_loss + l2_regu_rate * L2

        # Create list params and updates method with ADAM Optimizer
        PARAMS_LIST = (EMBD.params + INPUT_FUSION_LAYER.params +
                       EPISODIC_MEM_PASS1.params + EPISODIC_MEM_PASS2.params +
                       EPISODIC_MEM_PASS3.params + prediction_layer.params)

        UPDATE_PARAMS = ADAM_OPTIMIZER(loss=COST_VALUE,
                                       all_params=PARAMS_LIST,
                                       learning_rate=learning_rate)

        # Create function call for train and validate
        TRAIN = theano.function(
            inputs=[tensor_facts, tensor_question,
                    tensor_answers, train_state],
            outputs=[COST_VALUE, prediction_layer.cal_errors(true_label)],
            updates=UPDATE_PARAMS,
        )

        VALIDATE = theano.function(
            inputs=[tensor_facts, tensor_question,
                    tensor_answers, train_state],
            outputs=[NLL_loss, prediction_layer.cal_errors(true_label),
                     atten1, atten2, atten3],
            on_unused_input="warn"
        )

        def getAllParams():
            paramsTrained = {
                "EMBD": EMBD.get_params(),
                "INPUT_FUSION_LAYER": INPUT_FUSION_LAYER.get_params(),
                "EPISODIC_MEM_PASS1": EPISODIC_MEM_PASS1.get_params(),
                "EPISODIC_MEM_PASS2": EPISODIC_MEM_PASS2.get_params(),
                "EPISODIC_MEM_PASS3": EPISODIC_MEM_PASS3.get_params(),
                "prediction_layer": prediction_layer.get_params()
            }
            return(paramsTrained)
        min_error = 0.9
        ####################
        # TRAIN MODEL HERE #
        ####################
        for epc in range(n_epoch):
            print("################## New Epoch ##################")
            error_train = numpy.zeros(shape=(len(groups_data_train),))
            for group, data_train in enumerate(groups_data_train):
                n_inter = int(len(data_train) / batchsize) + 1
                list_error = []
                for iter_n in range(n_inter):
                    bz = min(batchsize, len(data_train))
                    sample_batch = get_batch(data_train, size=bz)
                    list_facts, list_ques, list_answ = sample_batch
                    cost, errors = TRAIN(list_facts, list_ques,
                                         list_answ, TRAINNING)
                    print("Epoch %i groups %i iter %i "
                          "with cost %f and errors %f"
                          % (epc, group, iter_n, cost, errors),
                          "input shape", numpy.array(list_facts).shape)

                paramsTrained = getAllParams()
                list_error.append(errors)
                error_train[group] = numpy.mean(list_error)

            if numpy.mean(error_train) < 0.9:
                error_val = []
                all_data_len = []
                for group, data_val in enumerate(groups_data_val):
                    data_len = len(data_val)
                    sample_validate = get_batch(data_val, size=data_len)
                    list_facts, list_ques, list_answ = sample_validate
                    cost, errors, at1, at2, at3 = VALIDATE(
                        list_facts, list_ques, list_answ, VALIDATING
                    )
                    print("################## VALIDATION ####################")
                    print("Epoch %i groups %i with cost %f and errors %f "
                          "in %i samples"
                          % (epc, group, cost, errors, data_len),
                          "input shape", numpy.array(list_facts).shape)
                    error_val.append(errors)
                    all_data_len.append(data_len)
                error_val = numpy.asarray(error_val)
                all_error = error_val * numpy.asarray(all_data_len)
                total_error = numpy.sum(all_error) / numpy.sum(all_data_len)
                print("Epoch %i with total error %f" % (epc, total_error))

                if total_error < min_error:
                    print("Save params with new error: %f" % total_error)
                    min_error = total_error
                    model_train = getAllParams()
                    with open(trainedParamsPath, 'wb') as handle:
                        pickle.dump(model_train, handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)
    # just for play
    else:
        TEST = theano.function(
            inputs=[tensor_facts, tensor_question, train_state],
            outputs=[prediction_layer.y_predict, atten1, atten2, atten3],
            on_unused_input="warn"
        )
        print("##############################################################")
        print("##############################################################")
        print("")
        print("Only use words in this list, other "
              "word might give different result", list(word_to_index.keys()))
        time.sleep(2)
        while True:
            sents = input("Please input evidence sentences separated by '; '"
                          " inpput 'END' for stop: ")
            if sents == "END":
                break
            ques = input("Please input question sentences: ")
            print("Parsing evidence sentences: ..... ")
            sents = sents.split("; ")
            indexed_sents = []
            for sent in sents:
                indexed_sent = []
                for word in sent.split(" "):
                    if word in word_to_index:
                        indexed_sent.append(word_to_index[word])
                    else:
                        indexed_sent.append(word_to_index["PADDING"])
                indexed_sents.append(indexed_sent)

            print("Parsing question sentences: ..... ")
            indexed_question = []
            for word in ques.split(" "):
                if word in word_to_index:
                    indexed_question.append(word_to_index[word])
                else:
                    indexed_question.append(word_to_index["PADDING"])

            max_words = max(map(len, indexed_sents))
            list_facts = padding(indexed_sents,
                                 shape=(len(indexed_sents), max_words))
            list_ques = indexed_question
            anws, att1, att2, att3 = TEST([list_facts], [list_ques], 0.0)

            print("The machine read story the first time and "
                  "gennerate the attention score for each "
                  "sentence as below: ...")
            print([str(round(elem, 3)) for elem in att1[0]])
            print("The machine read story the seccond time and "
                  "gennerate the attention score for each "
                  "sentence as below: ...")
            print([str(round(elem, 3)) for elem in att2[0]])
            print("The machine read story the third time and "
                  "gennerate the attention score for each "
                  "sentence as below: ...")
            print([str(round(elem, 3)) for elem in att3[0]])
            print("Then machine answer: ", index_to_word[str(anws[0]+1)])

if __name__ == "__main__":
#    prefix = "/mnt/sdd2/Projects/Replicate_Episodic_DynamicMemNet/Data/task_qa2/"
    prefix = sys.argv[1]
    mode = sys.argv[2]
    data_train = "train.npy"
    data_validate = "validate.npy"
    vocab_json = "vocab.json"
    model_path = "ModelResult/TrainedParams.pickle"
    main(data_train_path=prefix+data_train,
         data_validate_path=prefix+data_validate,
         vocabulary_path=prefix+vocab_json,
         trainedParamsPath=prefix+model_path,
         mode=mode)
