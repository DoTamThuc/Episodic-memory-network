#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:03:30 2017

@author: red-sky
"""

import numpy
import theano

import theano.tensor as T
from theano import config
from utils import createShareVar


class GRU_rnn(object):
    def __init__(self, RNG, num_in=80, num_out=160,
                 paramsTrained=None, name="GRU_rnn_"):
        # Init all variable in GRU cell:
        #    Wu, Uu, Bu : for update gate
        #    Wr, Ur, Br : for reset gate
        #    W, U, B : for state output

        # DETAIL Architecture is presented in :
        #   https://arxiv.org/pdf/1406.1078v3.pdf
        #   or page 3 in ref: https://arxiv.org/abs/1603.01417

        self.name = name
        self.n_in = num_in
        self.n_out = num_out

        # Init weight for update gate
        self.Wu = createShareVar(rng=RNG,
                                 dim=(num_in, num_out),
                                 name=name + "Wu",
                                 factor_for_init=num_in+num_out)
        self.Uu = createShareVar(rng=RNG,
                                 dim=(num_out, num_out),
                                 name=name + "Uu",
                                 factor_for_init=num_out+num_out)
        self.Bu = theano.shared(
            numpy.zeros(shape=(num_out, ), dtype=config.floatX),
            name=name + "Bu", borrow=True
        )

        # Init weight for reset gate
        self.Wr = createShareVar(rng=RNG,
                                 dim=(num_in, num_out),
                                 name=name + "Wr",
                                 factor_for_init=num_in+num_out)
        self.Ur = createShareVar(rng=RNG,
                                 dim=(num_out, num_out),
                                 name=name + "Ur",
                                 factor_for_init=num_out+num_out)
        self.Br = theano.shared(
            numpy.zeros(shape=(num_out, ), dtype=config.floatX),
            name=name + "Br", borrow=True
        )

        # Init weight for new state creation layers
        self.W = createShareVar(rng=RNG,
                                dim=(num_in, num_out),
                                name=name + "W",
                                factor_for_init=num_in+num_out)
        self.U = createShareVar(rng=RNG,
                                dim=(num_out, num_out),
                                name=name + "U",
                                factor_for_init=num_out+num_out)
        self.B = theano.shared(
            numpy.zeros(shape=(num_out, ), dtype=config.floatX),
            name=name + "B", borrow=True
        )
        self.params = [
            self.Wu, self.Uu, self.Bu,
            self.Wr, self.Ur, self.Br,
            self.W, self.U, self.B,
        ]
        if paramsTrained is not None:
            for p in self.params:
                print("Filling variable: ", p.name)
                p.set_value(paramsTrained[p.name])

        self.weight_list = [self.Wu, self.Uu, self.Wr, self.Ur, self.W, self.U]
        self.L2 = sum([(param**2).sum() for param in self.weight_list])

    # Get params for model saving
    def get_params(self):
        paramsTrained = {
            self.Wu.name: self.Wu.get_value(),
            self.Wr.name: self.Wr.get_value(),
            self.W.name: self.W.get_value(),
            self.Uu.name: self.Uu.get_value(),
            self.Ur.name: self.Ur.get_value(),
            self.U.name: self.U.get_value(),
            self.Bu.name: self.Bu.get_value(),
            self.Br.name: self.Br.get_value(),
            self.B.name: self.B.get_value(),
        }
        return(paramsTrained)

    # implementation of each step in GRU cell (loop)
    def _stepGRU(self, x, state):

        # DETAIL Architecture of each step is presented in :
        #   https://arxiv.org/pdf/1406.1078v3.pdf
        #   or page 3 in ref: https://arxiv.org/abs/1603.01417

        Ui = T.nnet.sigmoid(
            T.dot(x, self.Wu) +
            T.dot(state, self.Uu) +
            self.Bu
        )

        Ri = T.nnet.sigmoid(
            T.dot(x, self.Wr) +
            T.dot(state, self.Ur) +
            self.Br
        )

        Hi_ = T.tanh(
            T.dot(x, self.W) +
            Ri * T.dot(state, self.U) +
            self.B
        )

        Hi = Ui * Hi_ + (1 - Ui) * state

        # Treat problem with GPU memmory overflow
        # Hi = T.cast(Hi, dtype=config.floatX)
        return(Hi)

    # compute the oupput of the GRU layer
    # the scan funtion will scan through the sequence and produce new state
    # for next loop
    # the _stepGRU input is vectors seq and previous state
    def output(self, input_seq):
        shape_input = input_seq.shape
        bach_size, num_steps, n_in = shape_input

        states_input = input_seq.dimshuffle((1, 0, 2))
        output, update = theano.scan(
            fn=self._stepGRU,
            sequences=states_input,
            outputs_info=T.alloc(numpy.asarray(0.0, dtype=config.floatX),
                                 bach_size, self.n_out),
            n_steps=num_steps
        )
        output = output.dimshuffle((1, 0, 2))
        return(output)


class InputModule(object):
    def __init__(self, RNG_, input_sents, quesion,
                 num_in=80, num_out=160, name="IN", paramsTrained=None):
        if paramsTrained is None:
            paramsTrained = {
                "forward_GRU_": None,
                "backward_GRU_": None,
                "question_GRU_": None
            }

        # This is the implementation of Fusion Layer of the MODEL describe in
        #    https://arxiv.org/abs/1603.01417
        # It contains:
        # + forward_GRU: the forward pass of the input fact vectors sequence
        # + backward_GRU: the backward pass of the input fact vectors sequence
        # + Question Module: the GRU pass of the question word vectors sequence

        self.forward_GRU = GRU_rnn(
            RNG=RNG_, name="forward_GRU_",
            num_in=num_in, num_out=num_out,
            paramsTrained=paramsTrained["forward_GRU_"]
        )
        self.backward_GRU = GRU_rnn(
            RNG=RNG_, name="backward_GRU_",
            num_in=num_in, num_out=num_out,
            paramsTrained=paramsTrained["backward_GRU_"]
        )
        self.question_GRU = GRU_rnn(
            RNG=RNG_, name="question_GRU_",
            num_in=num_in, num_out=num_out,
            paramsTrained=paramsTrained["question_GRU_"]
        )

        self.forward_sents = self.forward_GRU.output(input_sents)
        self.backward_sents = self.backward_GRU.output(input_sents[:, ::-1, :])
        self.question = self.question_GRU.output(quesion)

        # the Bi-directional GRU
        #    the trick to understand here is that:
        #    "DON NOT FORGET TO REVERSE THE OUTPUT STATE OF BACKWARD PASS :D"

        self.bi_directional_sents = (self.forward_sents +
                                     self.backward_sents[:, ::-1, :])
        self.question_out = self.question[:, -1]

        # Create a list of params to be collected later
        self.params = (self.forward_GRU.params +
                       self.backward_GRU.params +
                       self.question_GRU.params)

        self.L2 = (self.forward_GRU.L2 +
                   self.backward_GRU.L2 +
                   self.question_GRU.L2)

        # Output of the bidirectinal F and question q
        self.output = [self.bi_directional_sents, self.question_out]

    def get_params(self):
        paramsTrained = {
            self.forward_GRU.name: self.forward_GRU.get_params(),
            self.backward_GRU.name: self.backward_GRU.get_params(),
            self.question_GRU.name: self.question_GRU.get_params()
        }
        return(paramsTrained)

if __name__ == "__main__":
    RNG_ = numpy.random.RandomState(220495)
    X1 = RNG_.uniform(size=(2, 4, 3))
    X2 = RNG_.uniform(size=(2, 6, 3))
    dataX1 = T.dtensor3("dataX1")
    dataX2 = T.dtensor3("dataX2")

    IN_MODULE = InputModule(RNG_, dataX1, dataX2, 3, 5)
    out_test = IN_MODULE.output

    func_test = theano.function(
        inputs=[dataX1, dataX2],
        outputs=out_test
    )
    result = func_test(X1, X2)
    print(result[0], result[0].shape)
    print(result[1], result[1].shape)
