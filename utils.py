from __future__ import print_function
from sys import getsizeof
from random import shuffle
import time
import math
import numpy as np

relative_path = './data/SetOf7604Proteins/'
trainList_addr = relative_path + 'trainList'
validList_addr = relative_path + 'validList'
testList_addr = relative_path + 'testList'

def read_list(filename):
    """Return a list of protein name
    """
    proteins_names = []
    with open(filename) as f:
        for line in f:
            protein_name = line.rstrip('\n')
            proteins_names.append(protein_name)
    return proteins_names

def read_protein(prot_name, relative_path, expand_dims=False):
    """Given a protein name, return a matrix of its features, corresponding
       second structure information, sequence length and mask.

    This is a sub-function of reading entire list of data and can also be
    used to read individual protein.
    """
    ss = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
    dict_ss = {key: value for (key, value) in zip(ss, range(8))}
    prot_addr = relative_path + '66FEAT/' + prot_name + '.66feat'
    ss_addr = relative_path + 'Angles/' + prot_name + '.ang'
    prot_mat = np.loadtxt(prot_addr)

    ss_mat = []
    with open(ss_addr) as f:
        next(f)
        for i, line in enumerate(f):
            line = line.split('\t')
            if line[0] == '0':
                ss_mat.append(dict_ss[line[3]])

    ss_mat = np.array(ss_mat).transpose()
    seq_len = ss_mat.shape[0]
    mask = np.ones((seq_len,), dtype=np.float32)

    if expand_dims:
        prot_mat = np.expand_dims(prot_mat, axis=0)
        ss_mat = np.expand_dims(ss_mat, axis=0)
        seq_len = np.array([seq_len])
        mask = np.expand_dims(mask, axis=0)

    return prot_mat, ss_mat, seq_len, mask


def generate_batch(prot_list, relative_path, max_seq_length=300, batch_size=4):
    """Given the list of protein name and length, generate a batch of feature 
       matrix and corresponding secondary structure label. Zero pad or cutoff 
       feature matrix based on max_seq_length.
    """
    while True:
        num_list = len(prot_list)
        batch_idx = np.random.randint(0, num_list, batch_size)

        # Create the batch matrix with all zeros for zero-padding.
        proteins_batch = np.zeros((batch_size, max_seq_length, 66),
                                  dtype=np.float32)
        ss_labels_batch = np.zeros((batch_size, max_seq_length),
                                   dtype=np.int32)
        mask_batch = np.zeros((batch_size, max_seq_length), dtype=np.int32)
        batch_seq_len = []

        for i, j in enumerate(batch_idx):
            protein_name = prot_list[j]
            protein_features, ss_labels, seq_len, mask = read_protein(protein_name, relative_path)
            min_idx = min(max_seq_length, seq_len)
            proteins_batch[i, :min_idx, :] = protein_features[:min_idx, :]
            ss_labels_batch[i, :min_idx] = ss_labels[:min_idx]
            mask_batch[i, :min_idx] = mask[:min_idx]
            batch_seq_len.append(min_idx)

        batch_seq_len = np.asarray(batch_seq_len, dtype=np.int32)
        yield proteins_batch, ss_labels_batch, batch_seq_len, mask_batch

def read_data(prot_list, relative_path, max_seq_length=300):
    """Given lists of training, validation and test datasets, return the 
       ndarrays of all data, labels and lengths.
    """
    num_list = len(prot_list)
    proteins_all = np.zeros((num_list, max_seq_length, 66), dtype=np.float32)
    ss_labels_all = np.zeros((num_list, max_seq_length), dtype=np.int32)
    mask_all = np.zeros((num_list, max_seq_length), dtype=np.float32)
    seq_lens_all = []

    for i, protein_name in enumerate(prot_list):
        protein_features, ss_labels, seq_len, mask = read_protein(protein_name, relative_path)
        min_idx = min(max_seq_length, seq_len)
        proteins_all[i, :min_idx, :] = protein_features[:min_idx, :]
        ss_labels_all[i, :min_idx] = ss_labels[:min_idx]
        mask_all[i, :min_idx] = mask[:min_idx]
        seq_lens_all.append(min_idx)
    seq_lens_all = np.asarray(seq_lens_all, dtype=np.int32)

    return proteins_all, ss_labels_all, seq_lens_all, mask_all

if __name__ == '__main__':
    trainList = read_list(trainList_addr)
    validList = read_list(validList_addr)
    testList = read_list(testList_addr)
    # print(len(validList))
    # print(validList)
    features, labels, seq_len, mask = read_protein(validList[0], elative_path, True)
    print(features.shape)
    print(labels.shape)
    print(type(seq_len))
    print(seq_len.shape)
    print(mask.shape)

    # proteins_all, ss_labels_all, seq_lens_all, mask_all = read_data(validList)
    # print(proteins_all.shape)
    # print(ss_labels_all.shape)
    # print(seq_lens_all)
    # print(mask_all.shape)

