#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:41:51 2017

@author: red-sky
"""
import numpy as np
import theano
from theano import config
from theano import tensor as T
import theano.typed_list


class EncodingLayer(object):
    def __init__(self, num_vocab, word_dim, rng, embedding_w=None):
        '''
        word_dim :: dimension of the word embeddings
        num_vocab :: number of word embeddings in the vocabulary
        embedding_w :: pre-train word vector
        '''
        # This is the implementation of input block of the MODEL described in:
        #    https://arxiv.org/abs/1603.01417
        # It contains:
        # + Words embedding vectors
        # + Positional Encoding Scheme in computation graph
        # check if embedding_w contain trained word embedding

        if embedding_w is None:
            word_vectors = np.asarray(
                    rng.uniform(-np.sqrt(3), np.sqrt(3),
                                (num_vocab, word_dim)),
                    dtype=config.floatX
            )
        else:
            word_vectors = np.asarray(
                    embedding_w["EmbeddingLayer_W"],
                    dtype=config.floatX
            )

        # Create shared variable for word embedding
        self.embedding_w = theano.shared(value=word_vectors,
                                         name="EmbeddingLayer_W",
                                         borrow=True)

        # Create shared vector for PADDING vectors - all zero
        self.PADDING = theano.shared(
                np.zeros(shape=(1, word_dim), dtype=config.floatX),
                name="EmbeddingLayer_PADDING", borrow=True
            )
        self.word_embedding = T.concatenate([self.PADDING, self.embedding_w],
                                            axis=0)

        # create list variable for optimization
        self.params = [self.embedding_w]
        self.infor = [num_vocab, word_dim]

    def get_params(self):
        # get parameters for saving
        paramsTrained = {
            self.embedding_w.name: self.embedding_w.get_value()
        }
        return(paramsTrained)

    def words_ind_2vec(self, index):
        # a simple embedding layers
        map_word_vectors = self.word_embedding[index]
        return map_word_vectors

    def positional_encoding_scheme(self, n_words):
        # positional encoding scheme layer described
        # in in Sukhbaatar et al. (2015) and page 3 in ref paper
        _, word_dim = self.infor
        result_mat = T.zeros(shape=(n_words, word_dim),
                             dtype=config.floatX)

        index1 = T.tile(T.arange(n_words), reps=word_dim)
        index2 = T.tile(T.arange(word_dim), reps=(n_words, 1))
        index2 = T.flatten(index2.dimshuffle(1, 0))

        def mini(j, d, A, n_words, word_dim):
            a = (1 - j / n_words) - (d / word_dim) * (1 - 2 * j / n_words)
            A = T.set_subtensor(A[j, d], a)
            return(A)

        output, _ = theano.scan(
            fn=mini,
            non_sequences=[n_words, word_dim],
            sequences=[index1, index2],
            outputs_info=[result_mat],
        )
        return(output[-1])

    def sents_ind_2vec(self, sents):
        # Create Input moddule contain positional encoding scheme
        # the input sents presents the index of words
        # this will convert each fact into a vector as output
        shape_input = sents.shape
        bach_size, n_sents, n_words = shape_input
        positional_encode_matrix = self.positional_encoding_scheme(n_words)
        p_e_m_shuffle = positional_encode_matrix.dimshuffle("x", "x", 0, 1)
        sents_emb = self.words_ind_2vec(sents) * p_e_m_shuffle
        return(sents_emb.sum(axis=2))

# Debug ONLY
if __name__ == "__main__":
    rng = np.random.RandomState(220495)
    arrSents = T.itensor3()
    nn = T.bscalar()
    EMBD = EncodingLayer(32, 10, rng=rng)
    Word2Vec = theano.function(
        inputs=[arrSents],
        outputs=EMBD.sents_ind_2vec(arrSents)
    )
    sents = [
        [[3, 14, 0],
         [0, 0, 0]],
        [[3, 14, 0],
         [1, 2, 6]]
    ]
    Vec = Word2Vec(sents)
    print("Val: ", Vec)
    print("Dim: ", Vec.shape)
