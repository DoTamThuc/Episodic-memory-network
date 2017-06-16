# Episodic-memory-network
Implementation of Dynamic Memory Networks in Theano introduced in https://arxiv.org/abs/1603.01417

The implementation contain three module:
 * Input Module: This module consist of some highlighted submodule
	- Word embedding layer
	- Positional encoding scheme
	- Input Fusion Layer (bi-directional GRU for sentences reader)
 * The Episodic Memory Module: This module consist of some highlighted submodule
	- Attention mechanism 
	- GRU plus: GRU base on attention mechanism
	- Episode memory updates
 * Answer Module
        
