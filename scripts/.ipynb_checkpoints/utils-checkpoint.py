# original python script written by Bertrand Martinez @ GoLP-EPP (2022)

import numpy as np
from matplotlib import pyplot as plt
from keras import models
from keras import layers
from tensorflow.keras.layers import BatchNormalization
import struct
from tqdm import tqdm

def set_seed():

    # This routine enables to set all the seeds to get reproducible results
    # with your python script on one computer's/laptop's CPU

    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = 42

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value)
    # for later versions: 
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K
    # for later versions:
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

def build_model_tcs(n_layers=2, n_nodes=4):
    model = models.Sequential()
    model.add (BatchNormalization(input_dim = 2))
    for i in range(n_layers):
        model.add (layers.Dense(n_nodes, activation="relu"))
    model.add (layers.Dense(1, activation="relu"))
    model.compile(optimizer = "rmsprop", loss='mse', metrics=["mape", "mse"])
    return model

def build_model_cdf(n_layers=2, n_nodes=4):
    model = models.Sequential()
    model.add (BatchNormalization(input_dim = 3))
    for i in range(n_layers):
        model.add (layers.Dense(n_nodes, activation="relu"))
    model.add (layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer = "rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
    return model

def prepare_targets_tcs(ds):
    _ds = ds.copy()
    M, N = _ds.shape

    _ds[:,0] = np.log10(_ds[:,0])
    _ds[:,1] = np.log10(_ds[:,1])

    _ds[:,N-1] = -np.log10(_ds[:,N-1])
    _ds[:,N-1] = (_ds[:,N-1]-min(_ds[:,N-1])) / (max(_ds[:,N-1])-min(_ds[:,N-1]))

    return _ds

def prepare_targets_cdf(ds):
    _ds = ds.copy()
    M, N = _ds.shape

    _ds[:,0] = np.log10(_ds[:,0])
    _ds[:,1] = np.log10(_ds[:,1])
    _ds[:,3] = _ds[:,3] - _ds[:,2]
    _ds[:,3] = (_ds[:,3]-min(_ds[:,3])) / (max(_ds[:,3])-min(_ds[:,3]))

    return _ds

def train_test_split(ds):

    # Get the dimensions
    _ds = ds.copy()
    M, N = _ds.shape

    # Shuffle the data
    np.random.shuffle(_ds)
    iX = np.arange(_ds.shape[0])
    np.random.shuffle(iX)
    _ds = _ds[iX]

    # Assign data and target
    data = _ds[:,0:N-1]
    targets = _ds[:,N-1]

    # Split train, validation, test (70%,15%,15%)
    train_split, val_split = 0.70, 0.85
    id1 = int(M*train_split)
    id2 = int(M*val_split)
    print("Train / Val / Test set: {0} / {1} / {2}".format(id1, id2-id1, M-id2))

    # data
    train_data = data[:id1]
    val_data = data[id1:id2]
    test_data = data[id2:]

    # target
    train_targets = targets[:id1]
    val_targets = targets[id1:id2]
    test_targets = targets[id2:]

    if np.isnan(np.min(targets)) == False:
        return train_data, train_targets, val_data, val_targets, test_data, test_targets

def plot_histo(y, bins,logscale):
    y = np.array(y)
    plt.hist(y, bins, color = 'indianred', alpha=0.5, label='Osiris')
    plt.legend(loc='upper right')
    plt.xlabel('target')
    plt.ylabel('number of occurrences')
    if logscale == 1:
        plt.yscale('log')
    plt.show()

def load_data(name):
    return np.loadtxt(name, delimiter=',')


def balance_data(dataset, nbins):

    M, N = dataset.shape
    y = dataset[:,-1]
    dy = (max(y) - min(y)) / y.shape
    edges = np.linspace(min(y)-dy/2., max(y)+dy/2., nbins+1)

    _ds = np.copy(dataset)
    # each of the nbins bin will have chunksize entries
    chunksize = int(M/nbins)
    # ordered list of numbers
    ilst = np.arange(0,M)
    for i in tqdm(range(nbins)):
        # get indices where y belongs to the i-th bin
        indexes = ilst[((y>=edges[i]) * (y<edges[i+1]))]
        if len(indexes) != 0:
            # selected random indices will be of size chunksize 
            indexes_balanced = np.random.choice(indexes, chunksize)
            # repeat chunksize times the entries of dataset
            _ds[i*chunksize:(i+1)*chunksize,:] = dataset[indexes_balanced,:]
    return _ds


def plot_history_tcs(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['mape']
    val_accuracy = history.history['val_mape']


    epochs = range(1, len(loss) + 1)
    fig, ax1 = plt.subplots()

    l1 = ax1.plot(epochs, loss, 'bo', label='Training loss')
    vl1 = ax1.plot(epochs, val_loss, 'b', label='Validation loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (mape)')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ac2= ax2.plot(epochs, accuracy, 'o', c="red", label='Training acc')
    vac2= ax2.plot(epochs, val_accuracy, 'r', label='Validation acc')
    ax2.set_ylabel('mape')
    ax2.set_yscale('log')

    lns = l1 + vl1 + vac2 + ac2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="center right")
    fig.tight_layout()
    fig.show()

def plot_history_cdf(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']


    epochs = range(1, len(loss) + 1)
    fig, ax1 = plt.subplots()

    l1 = ax1.plot(epochs, loss, 'bo', label='Training loss')
    vl1 = ax1.plot(epochs, val_loss, 'b', label='Validation loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (mape)')
#    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ac2= ax2.plot(epochs, accuracy, 'o', c="red", label='Training acc')
    vac2= ax2.plot(epochs, val_accuracy, 'r', label='Validation acc')
    ax2.set_ylabel('accuracy')
#    ax2.set_yscale('log')

    lns = l1 + vl1 + vac2 + ac2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="center right")
    fig.tight_layout()
    fig.show()

