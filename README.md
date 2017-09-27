# Protein Secondary Struture Prediction in TensorFlow

## Introduction
TensorFlow implementation of a protein secondary structure prediction network. In essence this is a variable sequence length
prediction so no prior knowledge of bioinformatics is needed to understand the code.

I don't think I am allowed to share the data but if you want to try out the code, you can download the cullpdb and cb513 
dataset from [this link](http://www.princeton.edu/~jzthree/datasets/ICML2014/), but you do need to modify the `utils.py` to
load the data.

## Dependencies
* Python 2.7
* TensorFlow 1.3 (>=1.0 should be fine for most part except the tf.argmax() function and RNNMultiCell())

## Files
* feat66.ipynb - initial and most experiments comes here. Everything is working except the last cell. CASP11 datase is not properly loaded yet.
* main.py - Run this to train the model.
* model.py - Restructured code for reusability and readability.
* utils.py - Preprocess the data.

## To do
1. Continue to improve the code structure.
2. Correctly incorporate batch normalization.
3. Enable command-line parsing for hyperparameter search.
4. Continue to improve the model.

## Acknowledgement
### RNN
* Initially I heavily referenced on [vyraun's code](https://github.com/vyraun/cb513-tensorflow).
* [Danijar Hafner's post](https://danijar.com/variable-sequence-lengths-in-tensorflow/) is super helpful in every aspect.
### Structuring Code
* [CS20si](https://web.stanford.edu/class/cs20si/).
* Check out [CS224n](http://web.stanford.edu/class/cs224n/) assignment 3 and 4 for structuring code. 
* [aicodes](https://github.com/aicodes/tf-bestpractice) is inspirational. It would be better if they post some simple examples.
