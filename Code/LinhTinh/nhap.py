#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:53:23 2017

@author: red-sky
"""

import numpy
import theano
from theano import config
import theano.tensor as T
RNG_ = numpy.random.RandomState(220495)


def createShareVar(rng, dim, name, factor_for_init):
    var_values = numpy.asarray(
        rng.uniform(
            low=-numpy.sqrt(6.0 / factor_for_init),
            high=numpy.sqrt(6.0 / factor_for_init),
            size=dim,
        )
    )
    Var = theano.shared(value=var_values, name=name, borrow=True)
    return Var


class GRU_rnn(object):
    def __init__(self, RNG, num_in=80, num_out=160,
                 paramsTrained=None, name="GRU_rnn_"):

        self.n_in = num_in
        self.n_out = num_out
        if paramsTrained is None:
            # Init weight for input gate
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

            # Init weight for output gate
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

            # Init weight for new state
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

    def _stepGRU(self, x, state):

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

        Hi = Ui * Hi_ + (1 - Ui) * Hi_

        return(Hi)

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


class Input_Module(object):
    def __init__(self, RNG_, input_sents, quesion,
                 num_in=80, num_out=160, name="IN"):
        self.forward_GRU = GRU_rnn(RNG=RNG_, name="forward_GRU_",
                                   num_in=num_in, num_out=num_out)
        self.backward_GRU = GRU_rnn(RNG=RNG_, name="backward_GRU_",
                                    num_in=num_in, num_out=num_out)

        self.question_GRU = GRU_rnn(RNG=RNG_, name="question_GRU_",
                                    num_in=num_in, num_out=num_out)

        self.forward_sents = self.forward_GRU.output(input_sents)
        self.backward_sents = self.backward_GRU.output(input_sents[:, ::-1, :])
        self.question = self.question_GRU.output(quesion)

        self.bi_directional_sents = (self.forward_sents +
                                     self.backward_sents[:, ::-1, :])
        self.question_out = self.question

        self.params = (self.forward_GRU.params +
                       self.backward_GRU.params +
                       self.question_GRU.params)

        self.output = [self.bi_directional_sents, self.question_out]


if __name__ == "__main__":
    X1 = RNG_.uniform(size=(2, 4, 3))
    X2 = RNG_.uniform(size=(2, 6, 3))
    dataX1 = T.dtensor3("dataX1")
    dataX2 = T.dtensor3("dataX2")

    IN_MODULE = Input_Module(RNG_, dataX1, dataX2, 3, 5)
    out_test = IN_MODULE.output

    func_test = theano.function(
        inputs=[dataX1, dataX2],
        outputs=out_test
    )
    result = func_test(X1, X2)
    print(result[0], result[0].shape)
    print(result[1], result[1].shape)





