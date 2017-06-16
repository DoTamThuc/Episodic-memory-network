# Episodic-memory-network
Implementation of Dynamic Memory Networks in Theano introduced in https://arxiv.org/abs/1603.01417

The implementation contain three module:
 * Input Module: with some highlighted submodules
	- Word embedding layer
	- Positional encoding scheme
	- Input Fusion Layer (bi-directional GRU for sentences reader)
 * The Episodic Memory Module: with some highlighted submodules
	- Attention mechanism 
	- GRU plus: GRU base on attention mechanism
	- Episode memory updates
 * Answer Module
        
