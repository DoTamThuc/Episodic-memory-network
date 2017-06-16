#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:40:35 2017

@author: red-sky
"""
import numpy
import theano
from theano import config
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def createShareVar(rng, dim, name, factor_for_init, method="Xavier"):
    print("Creating variable: ", name)
    factor_for_init = float(factor_for_init)
    if method == "uniform":
        var_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6.0 / factor_for_init),
                high=numpy.sqrt(6.0 / factor_for_init),
                size=dim,
            ),
            dtype=config.floatX
        )
    elif method == "Xavier":
        var_values = numpy.asarray(
            rng.normal(loc=0, scale=2 / factor_for_init, size=dim),
            dtype=config.floatX
        )
    Var = theano.shared(value=var_values, name=name, borrow=True)
    return Var


def padding(unequal_len_list, shape):
    len_list = len(unequal_len_list)
    new_list = []
    for f in unequal_len_list:
        len_f = len(f)
        pad_y = [0 for i in range(shape[1] - len_f)]
        new_list.append(f + pad_y)
    pad_x = [[0 for i in range(shape[1])] for j in range(shape[0] - len_list)]
    return(new_list + pad_x)


def get_batch(data, size=128):
    sampleIDs = range(len(data))
    if size == len(data):
        batch = data
    else:
        batch = data[numpy.random.choice(sampleIDs, size)]
    list_facts, list_ques, list_answ = [], [], []
    max_facts_len, max_words_len = 0, 0
    max_ques = 0
    for sample in batch:
        facts, ques, answ = sample
        max_ques = max(max_ques, len(ques))
        list_ques.append(ques)
        list_answ.append(answ)
        facts_len = len(facts)
        words_len = max([len(f) for f in facts])
        if (max_facts_len >= max(max_facts_len, facts_len) and
                max_words_len >= max(max_words_len, words_len)):
            new_facts = padding(facts, shape=(max_facts_len, max_words_len))
            list_facts.append(new_facts)
        else:
            max_facts_len = max(max_facts_len, facts_len)
            max_words_len = max(max_words_len, words_len)
            for i, facts_old in enumerate(list_facts):
                new_facts = padding(facts_old,
                                    shape=(max_facts_len, max_words_len))
                list_facts[i] = new_facts
            new_facts = padding(facts, shape=(max_facts_len, max_words_len))
            list_facts.append(new_facts)
    list_ques = padding(list_ques, shape=(len(list_ques), max_ques))
    return(list_facts, list_ques, list_answ)


def get_groups(Data, groups_len=5):
    len_of_facts = numpy.array([len(f) for f in Data[:, 0]])
    set_of_len = numpy.unique(len_of_facts)
    index_split = range(0, len(set_of_len), groups_len)
    groups = [set_of_len[x:x+groups_len] for x in index_split]
    Data_groups = []
    for group in groups:
        index = numpy.in1d(len_of_facts, group)
        Data_groups.append(Data[index])
    return(Data_groups)


class LogisticRegression(object):

    def __init__(self, rng, layerInput, n_in, n_out,
                 paramsLayer=None,
                 name="LogisticRegression_"):

        self.layerInput = layerInput
        if paramsLayer is None:
            self.W = createShareVar(rng=rng, name=name+"W",
                                    factor_for_init=n_out + n_in,
                                    dim=(n_in, n_out))
        else:
            print("Filling variables prediction_layer: W")
            self.W = theano.shared(value=paramsLayer["W"],
                                   name=name+"W", borrow=True)

        if paramsLayer is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values,
                                   name=name+"b", borrow=True)
        else:
            print("Filling variables prediction_layer: b")
            self.b = theano.shared(value=paramsLayer["b"],
                                   name=name+"b", borrow=True)

        step1 = T.dot(self.layerInput, self.W)
        self.prob_givenX = T.nnet.softmax(step1 + self.b)
        self.y_predict = T.argmax(self.prob_givenX, axis=1)

        self.params = [self.W]
        self.L2 = sum([(param**2).sum() for param in self.params])

    def get_params(self):
        trainedParams = {
            "W": self.W.get_value(),
            "b": self.b.get_value()
        }
        return(trainedParams)

    def neg_log_likelihood(self, y_true):
        y_true = T.cast(y_true, "int32")
        log_prob = T.log(self.prob_givenX)
        nll = -T.mean(log_prob[T.arange(y_true.shape[0]), y_true])
        return nll

    def cal_errors(self, y_true):
        if y_true.ndim != self.y_predict.ndim:
            raise TypeError(
                "y should have the same shape as self.y_pred",
                ("y_true", y_true.ndim, "y_pred", self.y_predict.ndim)
            )
        if y_true.dtype.startswith("int"):
            return T.mean(T.neq(self.y_predict, y_true))
        else:
            raise TypeError(
                "y_true should have type int ...",
                ("y_true", y_true.type, "y_pred", self.y_predict.type)
            )


def ADAM_OPTIMIZER(loss, all_params, learning_rate=0.001,
                   b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
    """
    CITE: http://sebastianruder.com/optimizing-gradient-descent/index.html#adam
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(numpy.float32(1))
    # (Decay the first moment running average coefficient)
    b1_t = b1*gamma**(t-1)

    for params_previous, g in zip(all_params, all_grads):
        init_moment = numpy.zeros(params_previous.get_value().shape,
                                  dtype=theano.config.floatX)
        # (the mean)
        first_moment = theano.shared(init_moment)
        # (the uncentered variance)
        second_moment = theano.shared(init_moment)

        # (Update biased first moment estimate)
        bias_m = b1_t*first_moment + (1 - b1_t)*g
        bias_m = T.cast(bias_m, dtype=config.floatX)

        # (Update biased second raw moment estimate)
        bias_v = b2*second_moment + (1 - b2)*g**2
        bias_v = T.cast(bias_v, dtype=config.floatX)

        # (Compute bias-corrected first moment estimate)
        unbias_m = bias_m / (1-b1**t)

        # (Compute bias-corrected second raw moment estimate)
        unbias_v = bias_v / (1-b2**t)

        # (Update parameters)
        update_term = (alpha * unbias_m) / (T.sqrt(unbias_v) + e)
        params_new = params_previous - update_term
        params_new = T.cast(params_new, dtype=config.floatX)

        updates.append((first_moment, bias_m))
        updates.append((second_moment, bias_v))
        updates.append((params_previous, params_new))
    updates.append((t, t + 1.))
    return updates


class DropOut(object):
    def __init__(self, RNG):

        # Generate a theano RandomStreams
        self.SRNG = RandomStreams(RNG.randint(220495))

    def drop(self, layer_input, keep_rate):
        mask = self.SRNG.binomial(n=1, p=keep_rate, size=layer_input.shape,
                                  dtype='floatX')
        output = theano.tensor.cast(layer_input * mask, theano.config.floatX)
        return(output)

    def dont_drop(self, layer_input, keep_rate):
        output = keep_rate * theano.tensor.cast(layer_input,
                                                theano.config.floatX)
        return(output)

    def output(self, layer_input, keep_rate, train):
        output = theano.ifelse.ifelse(
            condition=T.eq(train, 0.0),
            then_branch=self.dont_drop(layer_input, keep_rate),
            else_branch=self.drop(layer_input, keep_rate)
        )
        return(output)
