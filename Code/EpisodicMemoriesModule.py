#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:02:01 2017

@author: red-sky
"""

import numpy
import theano
from theano import config
import theano.tensor as T
from utils import createShareVar


class AttentionGate(object):
    def __init__(self, RNG, fact_dim=160, n_hiden=4*160, n_out=1,
                 paramsTrained=None, name="AttentionGate_"):
        # This is the implementation of the attention mechanism describe in:
        #    https://arxiv.org/abs/1603.01417:
        # This is just a simple feedforward network, where the input layer
        #    is the concation of different feature vector represents the
        #    interation of one given input fact i, the question, the previous
        #    memmory vector
        # The output is the score of the input fact i, represents it connection
        #    with the memmory and question

        #  Caution: Understand this is crucial for Episodic memmory network

        self.name = name
        self.W1 = createShareVar(
            rng=RNG,
            dim=(4 * fact_dim, n_hiden),
            name=name + "W1",
            factor_for_init=4 * fact_dim + n_hiden
        )
        self.B1 = theano.shared(
            numpy.asarray(RNG.normal(scale=0.01, size=(n_hiden, )),
                          dtype=config.floatX),
            name=name + "B1", borrow=True
        )

        self.W2 = createShareVar(
            rng=RNG,
            dim=(n_hiden, n_out),
            name=name + "W2",
            factor_for_init=n_hiden + n_out
        )
        self.B2 = theano.shared(
            numpy.asarray(RNG.normal(scale=0.01, size=(n_out, )),
                          dtype=config.floatX),
            name=name + "B2", borrow=True
        )

        self.n_out = n_out
        self.params = [self.W1, self.B1, self.W2, self.B2]
        if paramsTrained is not None:
            for p in self.params:
                print("Filling variable: ", p.name)
                p.set_value(paramsTrained[p.name])

        # L2 redulation
        self.weight_list = [self.W1, self.W2]
        self.L2 = sum([(param**2).sum() for param in self.weight_list])

    def get_params(self):
        paramsTrained = {
            self.W1.name: self.W1.get_value(),
            self.W2.name: self.W2.get_value(),
            self.B1.name: self.B1.get_value(),
            self.B2.name: self.B2.get_value(),
        }
        return(paramsTrained)

    def _stepAttention(self, fact, question, prev_mem):
        # Each step will score one input fact
        # the logic of each step is described clearly in:
        #    https://arxiv.org/abs/1603.01417

        list_feature = [
            fact * question,
            fact * prev_mem,
            T.abs_(fact - question),
            T.abs_(fact - prev_mem)
        ]
        concatenated_vec = T.concatenate(list_feature, axis=1)
        layer1 = T.tanh(T.dot(concatenated_vec, self.W1) +
                        self.B1.dimshuffle("x", 0))
        out_one = T.dot(layer1, self.W2) + self.B2.dimshuffle("x", 0)
        return(out_one)

    def output(self, list_facts, question, prev_mem):
        # Compute the output of the attention layer
        #    this will scan through the list facts and calculate
        #    score for each facts

        shape_input = list_facts.shape
        bach_size, num_facts, fact_dim = shape_input

        facts_input = list_facts.dimshuffle((1, 0, 2))
        output, update = theano.scan(
            fn=self._stepAttention,
            sequences=[facts_input],
            non_sequences=[question, prev_mem],
            n_steps=num_facts
        )
        output1 = output.dimshuffle((1, 0, 2))
        output = T.nnet.softmax(T.reshape(output1, (bach_size, num_facts)))
        return(output, T.reshape(output1, (bach_size, num_facts)))


class GRU_plus(object):
    def __init__(self, RNG, fact_dim=160, context_dim=160,
                 paramsTrained=None, name="GRU_plus_"):

        #  First thing to say:
        #    "THIS OBJECT IS DAMM INTERESTING TO BE CODED =))), I SPEND 1 DAYS,
        #     AND I LOVE EVERY MOMENT OF IT =)))"

        # It contains:
        # + The attention mechanism
        # + The GRU plus with the intergated attention mechanism

        # LET BEGIN THE FUN PART

        self.name = name
        self.n_in = fact_dim
        self.n_out = context_dim
        self.attention_name = name + "AttentionGate_"

        if paramsTrained is None:
            paramsTrained = {
                self.attention_name: None,
            }

        # Init the attention_gate of the GRU_plus
        self.attention_gate = AttentionGate(
            RNG, fact_dim=fact_dim, n_hiden=fact_dim, n_out=1,
            paramsTrained=paramsTrained[self.attention_name],
            name=self.attention_name
        )

        # Init weight for reset gate
        self.Wr = createShareVar(rng=RNG,
                                 dim=(fact_dim, context_dim),
                                 name=name + "Wr",
                                 factor_for_init=fact_dim+context_dim)
        self.Ur = createShareVar(rng=RNG,
                                 dim=(context_dim, context_dim),
                                 name=name + "Ur",
                                 factor_for_init=context_dim+context_dim)
        self.Br = theano.shared(
            numpy.zeros(shape=(context_dim, ), dtype=config.floatX),
            name=name + "Br", borrow=True
        )

        # Init weight for new state
        self.W = createShareVar(rng=RNG,
                                dim=(fact_dim, context_dim),
                                name=name + "W",
                                factor_for_init=fact_dim+context_dim)
        self.U = createShareVar(rng=RNG,
                                dim=(context_dim, context_dim),
                                name=name + "U",
                                factor_for_init=context_dim+context_dim)
        self.B = theano.shared(
            numpy.zeros(shape=(context_dim, ), dtype=config.floatX),
            name=name + "B", borrow=True
        )

        # Collect the trainable parameter of gru
        self.params_gru = [
            self.Wr, self.Ur, self.Br,
            self.W, self.U, self.B,
        ]

        for p in self.params_gru:
            if (p.name in paramsTrained):
                print("Filling variable: ", p.name)
                p.set_value(paramsTrained[p.name])

        # Add trainable parameter of attention_gate
        self.params = self.params_gru + self.attention_gate.params

        self.weight_list = [self.Wr, self.Ur, self.W, self.U]
        self.L2 = (sum([(param**2).sum() for param in self.weight_list]) +
                   self.attention_gate.L2)

    def get_params(self):
        paramsTrained = {
            self.attention_gate.name: self.attention_gate.get_params(),
            self.Wr.name: self.Wr.get_value(),
            self.W.name: self.W.get_value(),
            self.Ur.name: self.Ur.get_value(),
            self.U.name: self.U.get_value(),
            self.Br.name: self.Br.get_value(),
            self.B.name: self.B.get_value(),
        }
        return(paramsTrained)

    def getAttentionGate(self, list_facts, question, prev_mem):
        # Compute the attention gate for the list input facts
        result = self.attention_gate.output(list_facts, question, prev_mem)
        return(result)

    def _stepGRUplus(self, fact, attention_gate, state):
        # Input the pair Fact i and its attention score, compute the next state
        Ri = T.nnet.sigmoid(
            T.dot(fact, self.Wr) +
            T.dot(state, self.Ur) +
            self.Br
        )

        Hi_ = T.tanh(
            T.dot(fact, self.W) +
            Ri * T.dot(state, self.U) +
            self.B
        )
        attention_gate = attention_gate.dimshuffle(0, "x")
        Hi = attention_gate * Hi_ + (1 - attention_gate) * state
#        Hi = T.cast(Hi, dtype=config.floatX)
        return(Hi)

    def output(self, list_facts, question, prev_mem):
        # This compute the context vector each time the MODEL process the list
        #    fact, given previous memory vector and the question
        #    the context vector is the last state after the RNN net when it
        #    scan through the input fact
        # The contect vector will be used to update the memory vector in later
        #    this is the beautifull of the Episodic Memmory Network MODEL

        shape_input = list_facts.shape
        bach_size, num_fact, n_in = shape_input
        attention_gates, look = self.getAttentionGate(list_facts,
                                                      question, prev_mem)
        list_facts = list_facts.dimshuffle((1, 0, 2))
        attention_gates = attention_gates.dimshuffle((1, 0))
        output, update = theano.scan(
            fn=self._stepGRUplus,
            sequences=[list_facts, attention_gates],
            outputs_info=T.alloc(numpy.asarray(0.0, dtype=config.floatX),
                                 bach_size, self.n_out),
            n_steps=num_fact
        )
        context = output.dimshuffle((1, 0, 2))
        return(context[:, -1], attention_gates.dimshuffle((1, 0)))


class EpisodeMemoryUpdates(object):
    def __init__(self, RNG, fact_dim=160, context_dim=160, mem_dim=160,
                 paramsTrained=None, name="Episode_Memory_Updates_"):
        # This is the implememtation of the Memory Update layer
        #    the layer is a simple feedfordward network where:
        # + Input is the concation of the question vector, conext vector and
        #    previous memory vector
        # + Output is simple memory vector

        self.name = name
        dim = (fact_dim + context_dim + mem_dim, mem_dim)
        self.W = createShareVar(rng=RNG, dim=dim, name=name + "W",
                                factor_for_init=fact_dim+context_dim)

        self.B = theano.shared(
            numpy.zeros(shape=(mem_dim, ), dtype=config.floatX),
            name=name + "B", borrow=True
        )

        self.params = [self.W, self.B]
        if paramsTrained is not None:
            for p in self.params:
                print("Filling variable: ", p.name)
                p.set_value(paramsTrained[p.name])

        self.weight_list = [self.W]
        self.L2 = sum([(param**2).sum() for param in self.weight_list])

    def get_params(self):
        paramsTrained = {
            self.W.name: self.W.get_value(),
            self.B.name: self.B.get_value(),
        }
        return(paramsTrained)

    def output(self, prev_mem, context, question):
        list_feature = [
            prev_mem,
            context,
            question
        ]
        # like I said, a simple feed forward network
        concatenated_vec = T.concatenate(list_feature, axis=1)
        output = T.nnet.relu(T.dot(concatenated_vec, self.W) + self.B)
        return(output)


class EpisodicModule(object):
    def __init__(self, RNG, fact_dim=160, context_dim=160, mem_dim=160,
                 paramsTrained=None, name="Episodic_Module_"):
        # This is just an object that combines all part and small architectture
        #    of the Episodic Memory network module
        # the code here is simple, so let focus on above small and fun part :D

        self.name = name
        self.gru_plus_name = name+"@GRU_plus_"
        self.episodic_update_name = name+"@Episode_Memory_Updates_"

        if paramsTrained is None:
            paramsTrained = {
                self.gru_plus_name: None,
                self.episodic_update_name: None,
            }

        self.gru_plus_rnn = GRU_plus(
            RNG=RNG, fact_dim=fact_dim,
            context_dim=context_dim,
            name=self.gru_plus_name,
            paramsTrained=paramsTrained[self.gru_plus_name]
        )
        self.episodic_update = EpisodeMemoryUpdates(
            RNG=RNG, fact_dim=fact_dim,
            context_dim=context_dim, mem_dim=mem_dim,
            name=self.episodic_update_name,
            paramsTrained=paramsTrained[self.episodic_update_name]
        )
        self.params = self.gru_plus_rnn.params + self.episodic_update.params
        self.L2 = self.gru_plus_rnn.L2 + self.gru_plus_rnn.L2

    def get_params(self):
        paramsTrained = {
            self.gru_plus_rnn.name: self.gru_plus_rnn.get_params(),
            self.episodic_update.name: self.episodic_update.get_params()
        }
        return(paramsTrained)

    def output(self, list_facts, ques, prev_mem):
        context_C, atten = self.gru_plus_rnn.output(list_facts, ques, prev_mem)
        new_mem = self.episodic_update.output(prev_mem, context_C, ques)
        return(new_mem, atten)

# JUST FOR DEBUG THE MODULE, DO NOT READ
if __name__ == "__main__":
    RNG_ = numpy.random.RandomState(220495)
    bz, n_fact, fact_dim, c_dim, mem_dim = 4, 10, 80, 160, 80

    list_facts = RNG_.normal(size=(bz, n_fact, fact_dim))
    ques = RNG_.uniform(size=(bz, fact_dim))
    prev_mem = RNG_.uniform(size=(bz, mem_dim))

    tensor_list_facts = T.dtensor3("tensor_list_facts")
    tensor_ques = T.dmatrix("tensor_ques")
    tensor_prev_mem = T.dmatrix("tensor_prev_mem")

    object_1 = EpisodicModule(RNG_, fact_dim=fact_dim,
                              context_dim=c_dim, mem_dim=mem_dim,
                              name="EpisodicModule_1")
#    My_bject_2 = EpisodicModule(RNG_, fact_dim=fact_dim,
#                              context_dim=c_dim, mem_dim=mem_dim,
#                              name="EpisodicModule_2")
#    Name_object_3 = EpisodicModule(RNG_, fact_dim=fact_dim,
#                              context_dim=c_dim, mem_dim=mem_dim,
#                              name="EpisodicModule_3")
#    is_object_3 = EpisodicModule(RNG_, fact_dim=fact_dim,
#                              context_dim=c_dim, mem_dim=mem_dim,
#                              name="EpisodicModule_3")
#

    out_test_Q1, CCC = object_1.gru_plus_rnn.output(tensor_list_facts,
                                 tensor_ques, tensor_prev_mem)
#    Tam_out_test_2 = object_2.output(tensor_list_facts,
#                                 tensor_ques, out_test_1)
    out_test_Q2, BBB = object_1.gru_plus_rnn.output(tensor_list_facts,
                                 tensor_ques, tensor_prev_mem)
    out_test_1, AAA = object_1.gru_plus_rnn.output(tensor_list_facts,
                                 tensor_ques, tensor_prev_mem)
#    Thuc_out_test_3 = object_2.output(tensor_list_facts,
#                                 tensor_ques, out_test_2)

    func_test = theano.function(
        inputs=[tensor_list_facts, tensor_ques, tensor_prev_mem],
        outputs=AAA
    )
    for i in range(1):
        n_fact = RNG_.choice(range(2, 3), size=1)[0]
        list_facts = RNG_.normal(size=(bz, n_fact, fact_dim))
        ques = RNG_.normal(size=(bz, fact_dim))
        prev_mem = RNG_.normal(size=(bz, mem_dim))
        result = func_test(list_facts, ques, prev_mem)
        print("n_fact: ", n_fact, result[0], numpy.array(result).shape)
