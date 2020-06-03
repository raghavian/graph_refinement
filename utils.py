import numpy as np
from os.path import isfile
from os import rename
import scipy.sparse as sp
from pickle import load, dump
import gzip
import torch
#import matplotlib.pyplot as plt

SMOOTH=1

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def to_linear_idx(x_idx, y_idx, num_cols):
    assert num_cols > np.max(x_idx)
    x_idx = np.array(x_idx, dtype=np.int32)
    y_idx = np.array(y_idx, dtype=np.int32)
    return y_idx * num_cols + x_idx


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def unpackAdj(filename):
    "Unpack adjacency matrices"
    f = gzip.open(filename, 'rb')
    adj = load(f)
    return adj.reshape(adj.shape[1:])

def dice_loss(preds, labels):
    "Return dice score. "
    preds_sq = preds**2
    return 1 - (2. * (torch.sum(preds * labels)) + SMOOTH) / \
            (preds_sq.sum() + labels.sum() + SMOOTH)


def focalCE(preds, labels, gamma):
    "Return focal cross entropy"
    loss = -torch.mean( ( ((1-preds)**gamma) * labels * torch.log(preds) ) \
    + ( ((preds)**gamma) * (1-labels) * torch.log(1-preds) ) )
    return loss

def dice(preds, labels):
    "Return dice score"
    preds_bin = (preds > 0.5).type_as(labels)
    return 2. * torch.sum(preds_bin * labels) / (preds_bin.sum() + labels.sum())

def wBCE(preds, labels, w):
    "Return weighted CE loss."
    return -torch.mean( w*labels*torch.log(preds) + (1-w)*(1-labels)*torch.log(1-preds) )

def unpack(filename):
    """
    Unpack nodes, input and ouput adjacency matrices
    from the pickle.
    """
    f = gzip.open(filename, 'rb')
    nodes, input_adj, output_adj = load(f)
    return nodes, input_adj, output_adj

def unpackNpz(filename):
    """ Unpack nodes, ip and op adj matrices from npz"""
    data = np.load(filename)
    return data['nodes'], data['ipAdj'], data['opAdj']

def pack(filename, var):
    """
    Gzip + pickle var into filename.
    """
    with gzip.GzipFile(filename, 'w') as f:
        dump(var, f)

def makeLogFile(filename="lossHistory.txt"):
    if isfile(filename):
        rename(filename,"lossHistoryOld.txt")

    with open(filename,"w") as text_file:
        print('Epoch\tlossTr\tlTrStd\taccTr\taTrStd\tlossVl\tlVlStd\taccVl\taVlStd\ttime(s)',file=text_file)
    print("Log file created...")
    return

def writeLog(logFile, epoch, lossTr, lTrStd, accTr, aTrStd, lossVl, lVlStd, accVl, aVlStd, eTime):
    print('Epoch:{:04d}\t'.format(epoch + 1),
          'lossTr:{:.4f}\t'.format(lossTr),
          'accTr:{:.4f}\t'.format(accTr),
          'lossVl:{:.4f}\t'.format(lossVl),
          'accVl:{:.4f}\t'.format(accVl),
          'time:{:.4f}'.format(eTime))

    with open(logFile,"a") as text_file:
        print('{:04d}\t'.format(epoch + 1),
                '{:.4f}\t'.format(lossTr),
                '{:.4f}\t'.format(lTrStd),
                '{:.4f}\t'.format(accTr),
                '{:.4f}\t'.format(aTrStd),
                '{:.4f}\t'.format(lossVl),
                '{:.4f}\t'.format(lVlStd),
                '{:.4f}\t'.format(accVl),
                '{:.4f}\t'.format(aVlStd),
                '{:.4f}'.format(eTime),file=text_file)


def plotLearningCurve():
    plt.clf()
    tmp = np.load('loss_tr.npy')
    plt.plot(tmp,label='Tr.Loss')
    tmp = np.load('loss_vl.npy')
    plt.plot(tmp,label='Vl.Loss')
    tmp = np.load('dice_tr.npy')
    plt.plot(tmp,label='Tr.Dice')
    tmp = np.load('dice_vl.npy')
    plt.plot(tmp,label='Vl.Dice')
    plt.legend()
    plt.grid()
    plt.show()

