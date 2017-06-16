# Episodic-memory-network
Implementation of Dynamic Memory Networks in Theano introduced in https://arxiv.org/abs/1603.01417

I consider the implementation as a self learning process, the code was written as the time I was learning Theano. I finish it after 2 weeks. All the training process and experiment was conducted on my laptop with only CPU, I do not have a GPU to test it. 

But it is very fun to see the model worked as the paper reported. It achieved nearly 100% accurate on the first two task (1, 2) in bAbI tasks data. I am training the model on later task to see how it works, it takes lot of time since I use only CPU.

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

### Prerequisites

Things you need to install before go further

```
python 3.6
theano
json
pickle
numpy
```
CAUTION: 
* The parameters of model was saved in pickle format with python 3.6, so if you use python 2.7, you should train again the model using difference path.
* You should provide the permission for the shell bash code ** run.sh **

### Task 1

In task 1, the model have to answer where is the person given the list of facts, and there are 1 supporting fact for the answer. The model achieve 99.99% accuracy on this task.

Examples of the task:

```
1 John travelled to the hallway.
2 Mary journeyed to the bathroom.
3 Where is John? 	hallway	1
4 Daniel went back to the bathroom.
5 John moved to the bedroom.
6 Where is Mary? 	bathroom	2
7 John went to the hallway.
8 Sandra journeyed to the kitchen.
9 Where is Sandra? 	kitchen	8

```
Note: the list of facts can expand to 15

To play with the trained model just run the command:

```
./run.sh --task=qa1 --mode=PLAY

OUTPUT: 
Only use words in this list, other word might give different result: [u'hallway', u'bathroom', u'John', u'garden', u'office', u'is', u'Sandra', u'moved', u'back', u'Mary', u'PADDING', u'to', u'Daniel', u'bedroom', u'went', u'journeyed', u'Where', u'the', u'travelled', u'kitchen']

Please input evidence sentences separate by '; ' inpput 'END' for stop:
```

You can input the facts: 
```
John travelled to the hallway; Mary journeyed to the bathroom
```
Then, input question when "Please input question sentences:" pop up

```
Where is John
```

Then results are printed
```
Parsing evidence sentences: ..... 
Parsing question sentences: ..... 
The machine read story the first time and gennerate the attention score for each sentence as below: ...
['0.5', '0.5']
The machine read story the seccond time and gennerate the attention score for each sentence as below: ...
['0.998', '0.002']
The machine read story the third time and gennerate the attention score for each sentence as below: ...
['0.564', '0.436']
```

The task canbe train again using the command:

```
./run.sh --task=qa1 --mode=TRAIN --data_path=**your/preferred/model/path/here/**
```




