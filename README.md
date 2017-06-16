# Episodic-memory-network
Implementation of Dynamic Memory Networks in Theano introduced in https://arxiv.org/abs/1603.01417

I consider the implementation as a self learning process, the code was written as the time I was learning Theano. I finish it after 2 weeks. All the training process and experiment was conducted on my laptop with only CPU, I do not have a GPU to test it. But it is very fun to see the model worked as the paper reported. It achieved nearly 100% accurate on the first three task (1,2,3) in bAbI tasks data.

The implementation contain three module:
 * Input Module: with some highlighted submodules
	- Word embedding layer
	- Positional encoding scheme
	- Input Fusion Layer (bi-directional GRU for sentences reader)
 * The Episodic Memory Module: with some highlighted submodules
	- Attention mechanism: a small "switch" was written here to output the attention score for each sentence.
	- GRU plus: GRU base on attention mechanism
	- Episode memory updates
 * Answer Module

## Getting Started

This will describe each task and show how to run the model on each task.


