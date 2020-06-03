import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
from itertools import combinations
import time
from pdb import set_trace
import argparse

import numpy as np
#from scipy.spatial.distance import squareform, pdist, cdist
from scipy.special import comb 
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

from utils import *

EPS = 1e-16

def weightedBCE(output, target, weight):

    return -torch.mean((1/weight)*target*torch.log(output+EPS)+\
			(weight)*(1-target)*torch.log(1-output+EPS))

def binary_accuracy(output, labels):

        preds = output > 0.5
        correct = preds.type_as(labels).eq(labels).double()
        correct = correct.sum()

        return correct / (len(labels)**2)


class mfaLayer(torch.nn.Module):

    def __init__(self):

        super(mfaLayer, self).__init__()
        self.b0 = nn.Parameter(1e0*(torch.rand(1)-0))
        self.b1 = nn.Parameter(1e0*(torch.rand(1)-0))
        self.b2 = nn.Parameter(1e0*(torch.rand(1)-0))
        self.lam = nn.Parameter(1e0*(torch.rand(1)-0))
        self.eta = nn.Parameter(1e0*(torch.rand(1)-0))
        self.nu = nn.Parameter(1e0*(torch.rand(1)-0))
        self.mu = nn.Parameter(1e0*(torch.rand(1)-0))
        self.kap = nn.Parameter(1e0*(torch.rand(1,deg-2)-0))

    def forward(self,X,T,b):
        #set_trace()
        gamma = Variable(torch.zeros((N,N)).cuda(args.gpu))
        X_ = (1-X+EPS)
        XT = torch.transpose(X,0,1)
        X3D = X.repeat(N,1,1).transpose(0,1)        

        lX_ = torch.log(X_)
        tmp0 = (torch.sum(lX_,1))
        tmp0 = (torch.exp(tmp0.repeat(N,1).transpose(0,1)-lX_))

        XX_ = X/X_
        XX_3D = XX_.repeat(N,1,1).transpose(0,1)
        locUp0 = (torch.sum(XX_3D.gather(2,idx_ij[:,:,:,b]),2))

        XX_4D =(XX_.repeat(num_comb,1,1).transpose(0,1)).repeat(N,1,1,1).transpose(0,1) 
        locUp1 = (torch.sum(XX_3D.gather(2,idx_ijF[:,:,:,b])*torch.sum(XX_4D.gather(3,idx_ijT[:,:,:,:,b]),3),2))
        
        gamma = (self.b1-self.b0+ (self.b2-self.b1)*locUp0 \
                -self.b2*locUp1)*tmp0        
        gamma += (4*XT-2)*(self.lam + self.eta*D[:,:,b] + self.nu*C[:,:,b]+self.mu*S[:,:,b])+100*(mask[:,:,b]-1) + torch.sum((4*X3D-2).gather(2,idx_ij[:,:,:,b])*self.kap,2)
        del XX_4D, XX_3D 
        XX_3D = XX_.repeat(num_comb_i,1,1).transpose(0,1)
        ## ELBO calculations

        elb = (self.b0+(torch.sum(XX_,1)*self.b1))
        elb+= torch.sum(torch.prod(XX_3D.gather(2,idx_iT[:,:,:,b]),2),1)*self.b2
        elb = torch.sum(elb*(torch.exp(torch.sum(lX_,1))))
        elb+= torch.sum((self.lam+self.eta*D[:,:,b]+self.nu*C[:,:,b]+self.mu*S[:,:,b])*(1+4*X*XT-2*(X+XT))*mask[:,:,b])
        elb+= torch.sum(torch.sum((1-2*(X3D.gather(2,idx_ij[:,:,:,b])+
            X3D[:,0:deg-2,:].transpose(1,2)) + 4*(X3D.gather(2,idx_ij[:,:,:,b])*(X3D[:,0:deg-2,:].transpose(1,2))))*self.kap,2)*mask[:,:,b])
        elb+= -T*torch.sum((X*torch.log(X+EPS)+X_*lX_)*mask[:,:,b])
        del X3D, lX_
        return elb, torch.sigmoid(gamma/T)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='/home/raghav/lung/data/DLCST_reference/cropped_data/train/gcn/batch1/volume3.pgz',
                    help='Path to training data file.')
parser.add_argument('--layers', type=int, default=5,
                    help='Number of MFA iterations(layers)/epoch .')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--nodes', type=int, default=100,
                    help='Number of nodes to process.')
parser.add_argument('--batch', type=int, default=1,
                    help='Number of batches of nodes.')
parser.add_argument('--degree', type=int, default=5,
                    help='Degree per node')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU ID, if CUDA is in use')
parser.add_argument('--temp', type=float, default=1.0,
                    help='Temperature for annealing')

args = parser.parse_args()

np.random.seed(42)
torch.manual_seed(42)
t0 = time.time()

nodesIn, X, labels = unpack(args.dataset)

#Parameters
cuda = 1
N = args.nodes
off = 0
epochs = args.epochs
deg = args.degree
batch = args.batch
Y_ = np.zeros((N,N,batch))
nodes = np.zeros((N,nodesIn.shape[0],batch))
D = np.zeros((N,N,batch))
S = np.zeros((N,N,batch))
C = np.zeros((N,N,batch))
Mnp = np.zeros((N,deg-1,batch),dtype=int)
mask = np.zeros((N,N,batch))
#Create all i,j permutations
num_comb = int(comb(deg-2,deg-3))
num_comb_i = int(comb(deg-1,2))
idx_ijT = np.zeros((N,N,num_comb,deg-3,batch), dtype=int)
idx_ij = np.zeros((N,N,deg-2,batch),dtype=int)
idx_ijF = np.zeros((N,N,deg-2,batch),dtype=int)
idx_iT = np.zeros((N,num_comb_i,2,batch),dtype=int)


for b in range(batch):
    Y_[:,:,b] = labels[b*N:(b+1)*N, b*N:(b+1)*N]
    nodes[:,:,b] = nodesIn[:,b*N:(b+1)*N].T

    #Compute Euclidean distance
    D[:,:,b] = (pairwise_distances(nodes[:,:3,b], metric='euclidean'))

    #Scale difference metric
    S[:,:,b] = 1/(1+pairwise_distances(np.reshape(nodes[:,3,b],(N,1)), 
        metric='l1'))

    #Compute Cosine similarity
    C[:,:,b] = 0.5*(1+cosine_similarity(nodes[:,4:,b]))

    Mnp[:,:,b] = (np.argsort(D[:,:,b],axis=1)[:,1:deg])
    for i in range(0,N):
        mask[i,Mnp[i,:,b],b] = 1
        idx = Mnp[i,:,b] 
        idx_iT[i,:,:,b] = (np.asarray \
                    (list(combinations(idx,2))))
        for j in idx:
            idx_ij[i,j,:,b] =  idx[idx != j] 
            idx_ijF[i,j,:,b] = idx[idx !=j][::-1]
            idx_ijT[i,j,:,:,b] = (np.asarray \
                    (list(combinations(idx_ij[i,j,:,b],deg-3))))

Y_= Variable(torch.FloatTensor(Y_))
Y = Variable(torch.zeros(N,N))
X0 = Variable(torch.ones(N,N,batch))

#Number of layers in the MFNN
#Corresponds to MFA iterations
num_layers = args.layers

mask = Variable(torch.FloatTensor(mask))
D = Variable(torch.FloatTensor(1/(1+D)),requires_grad=False)
C = Variable(torch.FloatTensor(C),requires_grad=False)
S = Variable(torch.FloatTensor(S),requires_grad=False)
T = Variable(torch.pow(args.temp,torch.range(num_layers-1,0,-1)),
        requires_grad=False)
M = Variable(torch.LongTensor(Mnp),requires_grad=False)
X0 = X0*mask

idx_ijT = Variable(torch.from_numpy(idx_ijT),requires_grad=False)
idx_ij = Variable(torch.from_numpy(idx_ij),requires_grad=False)
idx_ijF = Variable(torch.from_numpy(idx_ijF),requires_grad=False)
idx_iT = Variable(torch.from_numpy(idx_iT),requires_grad=False)

module = mfaLayer()

if(cuda):
    module.cuda(args.gpu)
    Y_ = Y_.cuda(args.gpu)
    X0 = X0.cuda(args.gpu)
    Y = Y.cuda(args.gpu)
    M = M.cuda(args.gpu)
    mask = mask.cuda(args.gpu)
    idx_ijT = idx_ijT.cuda(args.gpu)
    idx_ij = idx_ij.cuda(args.gpu)
    idx_ijF = idx_ijF.cuda(args.gpu)
    idx_iT = idx_iT.cuda(args.gpu)
    D = D.cuda(args.gpu)
    C = C.cuda(args.gpu)
    S = S.cuda(args.gpu)
    T = T.cuda(args.gpu)

optimizer = optim.Adam(module.parameters(),lr=1e-1)
loss_train = np.zeros(epochs)
acc = np.zeros(epochs)
param = np.zeros((7+deg-2,epochs))
grad = np.zeros((7,epochs))
elbo = Variable(torch.zeros((epochs,num_layers)))

set_trace()
for epoch in range(epochs):
    t = time.time()
    # Zero gradients
    optimizer.zero_grad()
    loss = 0.0
    acc_train = 0.0
    sequence = np.random.permutation((batch))
    for b in sequence:
        X = X0[:,:,b]
        # Forward pass: Compute predicted y by passing x to the model
        for l in range(num_layers):
            elbo[epoch,l], Y = module(X,T[l],b)
            X = Y

        # Compute and print loss
        loss += F.binary_cross_entropy(Y, Y_[:,:,b])
        acc_train += binary_accuracy(Y,Y_[:,:,b])
        # Zero gradients, perform a backward pass, and update the weights.
        loss.backward(retain_graph=True)
    optimizer.step()
    acc[epoch] = acc_train/(batch)
    loss_train[epoch] = loss/(batch)

    if(not(np.mod(epoch+1,10))):
        print('Epoch:{:04d}'.format(epoch+1), 
                'loss_train:{:.4f}'.format(loss_train[epoch]),
                'acc_train:{:.4f}'.format(acc[epoch]),
                'time: {:.4f}s'.format(time.time() - t))
    param[0,epoch] = module.b0.cpu().data.numpy()
    param[1,epoch] = module.b1.cpu().data.numpy()
    param[2,epoch] = module.b2.cpu().data.numpy()
    param[3,epoch] = module.lam.cpu().data.numpy()
    param[4,epoch] = module.eta.cpu().data.numpy()
    param[5,epoch] = module.nu.cpu().data.numpy()
    param[6,epoch] = module.mu.cpu().data.numpy()
    param[7:,epoch] = module.kap.cpu().data.numpy()
print('Optimisation done. Total time {:.4f}s'.format(time.time()-t0))
"""
    """

"""
    grad[0,epoch] = module.b0.grad.data.numpy()
    grad[1,epoch] = module.b1.grad.data.numpy()
    grad[2,epoch] = module.b2.grad.data.numpy()
    grad[3,epoch] = module.lam.grad.data.numpy()
    grad[4,epoch] = module.eta.grad.data.numpy()
    grad[5,epoch] = module.nu.grad.data.numpy()
    grad[6,epoch] = module.mu.grad.data.numpy()"""

