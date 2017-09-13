from __future__ import print_function
from sys import getsizeof
import numpy as np
import tensorflow as tf
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

trainList_addr = './data/trainList'
validList_addr = './data/validList'
testList_addr = './data/testList'

def read_list(filename):
    """Return a list of protein name

    Args:
        filename: the path of train, validation or test file.

    Returns:
        A list of proteins' names specified in each file.
    """
    proteins_name = []
    proteins_length = []
    with open(filename) as f:
        for line in f:
            protein_name = line.rstrip('\n')
            prot_addr = './data/66FEAT/' + protein_name + '.66feat'
            protein_length = 0
            with open(prot_addr) as fs:
                for i, l in enumerate(fs):
                    pass
            protein_length = i + 1
            proteins_name.append(protein_name)
            proteins_length.append(protein_length)
    # print("List length is {}".format(len(ls)))

    return proteins_name, proteins_length

def read_protein(prot_name, test=False):
    """Given a protein name, return a matrix of its features and corresponding
       second structure information

    Args:
        prot_name: The protein name provided by list of proteins from 
            read_list(). 

    Returns:
        The numpy array of residue features and secondary structure label.
    """
    ss = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
    dict_ss = {key: value for (key, value) in zip(ss, range(8))}
    prot_addr = './data/66FEAT/' + prot_name + '.66feat'
    ss_addr = './data/Angles/' + prot_name + '.ang'
    prot_mat = np.loadtxt(prot_addr)

    ss_mat = []
    with open(ss_addr) as f:
        next(f)
        for i, line in enumerate(f):
            line = line.split('\t')
            if line[0] == '0':
                if(test):
                    ss_mat.append(line[3])
                else:
                    ss_mat.append(dict_ss[line[3]])

    ss_mat = np.array(ss_mat).transpose()
    return prot_mat, ss_mat

def generate_batch(prot_list, prot_len_list, max_seq_length=300, batch_size=4):
    """Given the list of protein name and length, generate a batch of feature 
       matrix and corresponding secondary structure label. Zero pad or cutoff 
       feature matrix based on max_seq_length.

    Args:
        prot_list: list of protein names provided by read_list() function.
        prot_len_list: list of protein lengths in the same order of prot_list.
        max_seq_length: cutoff length for protein sequences.
        batch_size:

    Returns:
        a generator that generates zero-padded/cutoff residue feature matrix, 
        secondary sturcture labels and a list of actual length for each protein.
    """
    while True:
        num_list = len(prot_list)
        batch_idx = np.random.randint(0, num_list, batch_size)

        # Create the batch matrix with all zeros for zero-padding.
        proteins_batch = np.zeros((batch_size, max_seq_length, 66),
                                  dtype=np.float32)
        ss_labels_batch = np.zeros((batch_size, max_seq_length),
                                   dtype=np.int32)
        batch_seq_len = []

        for i, j in enumerate(batch_idx):
            protein_name = prot_list[j]
            protein_features, ss_labels = read_protein(protein_name)
            min_idx = min(max_seq_length, protein_features.shape[0])
            proteins_batch[i, :min_idx, :] = protein_features[:min_idx, :]
            ss_labels_batch[i, :min_idx] = ss_labels[:min_idx]
            # batch_seq_len.append(prot_len_list[j])
            batch_seq_len.append(min_idx)

        batch_seq_len = np.asarray(batch_seq_len, dtype=np.int32)
        yield proteins_batch, ss_labels_batch, batch_seq_len

def read_data(prot_list, prot_len_list, max_seq_length=300):
    """Given lists of training, validation and test datasets, return the 
       ndarrays of all data, labels and lengths.
    """
    num_list = len(prot_list)
    proteins_all = np.zeros((num_list, max_seq_length, 66), dtype=np.float32)
    ss_labels_all = np.zeros((num_list, max_seq_length), dtype=np.int32)
    seq_lens_all = []

    for i, protein_name in enumerate(prot_list):
        protein_features, ss_labels = read_protein(protein_name)
        min_idx = min(max_seq_length, protein_features.shape[0])
        proteins_all[i, :min_idx, :] = protein_features[:min_idx, :]
        ss_labels_all[i, :min_idx] = ss_labels[:min_idx]
        seq_lens_all.append(min_idx)
    seq_lens_all = np.asarray(seq_lens_all, dtype=np.int32)

    return proteins_all, ss_labels_all, seq_lens_all


# Define test functions for some data preprocessing functions
def test_read_protein():
    test_protein = trainList[2][0]
    print("Test protein name is {}".format(test_protein))
    prot, ss = read_protein(test_protein, test=True)
    print("Protein feature dimension is {}".format(prot.shape))
    print("Second structure information dimension is {}".format(ss.shape))
    print(ss)

def test_generate_batch():
    generator = generate_batch(trainList, train_len_list)
    prot, ss_label, batch_seq_len = generator.next()

    print(batch_seq_len)
    print(prot.shape)
    print(ss_label.shape)
    print('_'*100)
    # print(ss_label[1,:])

def plot_length_distribution():
    print("There are {} proteins in the training set" \
          .format(len(train_len_list)))
    # print(max(trainList, key=lambda item:item[1]))
    count = 0
    for i in train_len_list:
        if i >= 400:
            count += 1
    print(count)
    plt.hist(train_len_list, bins=50)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    trainList, train_len_list = read_list(trainList_addr)
    validList, valid_len_list = read_list(validList_addr)
    testList, _ = read_list(testList_addr)

    # test_read_protein()
    # plot_length_distribution()
    # test_generate_batch()

    # generator = generate_batch(trainList, train_len_list)
    # print(generator.next()[0].shape)
    # print(generator.next()[1].shape)
    # print(generator.next()[2])

    a, b, c = read_data(validList, valid_len_list)
    print(a.shape, b.shape, c.shape)
