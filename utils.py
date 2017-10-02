from __future__ import print_function
import numpy as np


def read_list(filename):
    """Read the protein list file and return a list of protein name"""
    proteins_names = []
    with open(filename) as f:
        for line in f:
            protein_name = line.rstrip('\n')
            proteins_names.append(protein_name)
    return proteins_names


def get_proteins_length(proteins_list, relative_path):
    """Given a list of proteins, return the lengths for each protein."""
    proteins_length = []
    for protein_name in proteins_list:
        protein_addr = relative_path + '66FEAT/' + protein_name + '.66feat'
        with open(protein_addr) as f:
            for i, l in enumerate(f):
                pass
            protein_length = i + 1
            proteins_length.append(protein_length)

    return np.array(proteins_length)


def read_protein(protein_name, relative_path, max_seq_len=300, padding=False):
    """Given a protein name, return the ndarray of features [1 x seq_len x n_features]
    and labels [1 x seq_len].
    """
    ss = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
    dict_ss = {key: value for (key, value) in zip(ss, range(8))}
    features_addr = relative_path + '66FEAT/' + protein_name + '.66feat'
    labels_addr = relative_path + 'Angles/' + protein_name + '.ang'

    protein_features = np.loadtxt(features_addr)
    protein_labels = []
    with open(labels_addr) as f:
        next(f)
        for i, line in enumerate(f):
            line = line.split('\t')
            if line[0] == '0':
                # 0 means the current ss label exists.
                protein_labels.append(dict_ss[line[3]])
    protein_labels = np.array(protein_labels).transpose()
    if padding:
        # if features passes max_seq_len, cutoff
        if protein_features.shape[0] >= max_seq_len:
            protein_features = protein_features[:max_seq_len, :]
            protein_labels = protein_labels[:max_seq_len]
        # else, zero-pad to max_seq_len
        else:
            padding_length = max_seq_len - protein_features.shape[0]
            protein_features = np.pad(protein_features, ((0, padding_length), (0, 0)),
                                      'constant', constant_values=((0, 0), (0, 0)))
            protein_labels = np.pad(protein_labels, (0, padding_length), 'constant', constant_values=(0, 0))

    protein_features = np.expand_dims(protein_features, axis=0)
    protein_labels = np.expand_dims(protein_labels, axis=0)

    return protein_features, protein_labels


def minibatches(protein_list, relative_path, batch_size, max_seq_len=None, shuffle=True):
    """Given a list of proteins' name and length, return a generateor of batch 
    of protein features and labels, zero-padded to the longest sequence within 
    the batch.
    """
    data_size = len(protein_list)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for batch_start in np.arange(0, data_size, batch_size):
        batch_indices = indices[batch_start : (batch_start + batch_size)]
        yield get_batch(protein_list, relative_path, batch_indices, max_seq_len)


def get_batch(protein_list, relative_path, batch_indices, max_seq_len=None):
    """Given indices of the desired protein, return the batched and zero-padded
    features [batch_size x max_seq_len x n_features] and 
    labels [batch_size x max_seq_len].
    If max_seq_len is given, zero-pad to max_seq_len, otherwise zero-pad to the
    longest sequence within the batches.
    """
    proteins_length = get_proteins_length(protein_list, relative_path)
    batch_names = np.array(protein_list)[batch_indices]
    if not max_seq_len:
        max_seq_len = max(proteins_length[batch_indices])

    protein_features, protein_labels = read_protein(batch_names[0], relative_path, max_seq_len, padding=True)
    for protein_name in batch_names[1:]:
        feature, label = read_protein(protein_name, relative_path, max_seq_len, padding=True)
        # print(feature.shape)
        # print(protein_features.shape)
        protein_features = np.concatenate([protein_features, feature]) 
        protein_labels = np.concatenate([protein_labels, label]) 

    return protein_features, protein_labels


##################
# Test Functions #
##################
def test_read_protein(protein_list, relative_path):
    test_protein = protein_list[0]
    features, labels = read_protein(test_protein, relative_path, padding=False)
    print("Protein Name:", test_protein)
    print("features:", features.shape)
    print("labels:", labels.shape)


def test_get_batch(protein_list, relative_path):
    batch_indices = [0, 4, 5, 6]
    pf, pl = get_batch(protein_list, relative_path, batch_indices)
    print(pf.shape)
    print(pl.shape)


def test_minibatches(protein_list, relative_path):

    for i, batch in enumerate(minibatches(protein_list, relative_path, 128)):
        features, labels = batch
        print(features.shape, labels.shape)

if __name__ == '__main__':
    SetOf7604Proteins_path = '../data/SetOf7604Proteins/'
    trainList_addr = SetOf7604Proteins_path + 'trainList'
    validList_addr = SetOf7604Proteins_path + 'validList'
    testList_addr = SetOf7604Proteins_path + 'testList'

    casp11_path = '../data/CASP11/'
    casp11List_addr = casp11_path + 'proteinList'
    casp12_path = '../data/CASP12/'
    casp12List_addr = casp12_path + 'proteinList'

    trainList = read_list(trainList_addr)
    validList = read_list(validList_addr)
    testList = read_list(testList_addr)
    casp12List = read_list(casp12List_addr)

    print(get_proteins_length(validList, SetOf7604Proteins_path))
    test_read_protein(validList, SetOf7604Proteins_path)
    test_get_batch(validList, SetOf7604Proteins_path)
    test_minibatches(validList, SetOf7604Proteins_path)

