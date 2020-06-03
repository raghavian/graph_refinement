from __future__ import division
from __future__ import print_function

import time
import argparse

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
from lungDataset import *
from modules import EdgeGNN
import pdb

gamma = 0.5 # Gamma for Focal CE

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='gnn',
                    help='Path to training data file.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU number.')
parser.add_argument('--batch', type=int, default=1,
                    help='Size of mini-batch.')
parser.add_argument('--train_num', type=int, default=0,
                    help='Size of training set.')
parser.add_argument('--model', type=str,
                    help='Path to pre-trained model.')
parser.add_argument('--preTrain', action='store_true', default=False,
                    help='Use Pretrained model')
parser.add_argument('--save', action='store_true', default=False,
                    help='Save trained model')
parser.add_argument('--weight', type=float, default=0.5,
                    help='Weighting for wBCE.')
parser.add_argument('--niter', type=int, default=1,
                    help='Number of rec. GNN layers')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
mini = args.batch
weight = args.weight 

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#torch.backends.cudnn.enabled = False

# Load data and normalize
# Pre-processed data will be stored as a dict


print("Loading and preprocessing training data...")
#pdb.set_trace()
train_data_loader = lungDataset(root_dir=args.dataset + "/train/npz",
                                cuda=args.cuda, gpu=args.gpu)
num_train = len(train_data_loader)
if(args.train_num):
	num_train = args.train_num
train_data = [train_data_loader[i] for i in range(num_train)]
print("Loading and preprocessing validation data...")
val_data_loader = lungDataset(root_dir=args.dataset + "/validation/npz",
                              cuda=args.cuda, gpu=args.gpu)
val_data = [val_data_loader[i] for i in range(len(val_data_loader))]

# Create lossHistory.txt
logFile = time.strftime("%Y%m%d_%H_%M")+'.txt'

makeLogFile(logFile)


feature_shape = train_data[0]['features'].shape[1]
print("Training model...")

# Model and optimizer
model = EdgeGNN(nfeat=feature_shape,
                nhid=args.hidden,
                dropout=args.dropout,
                niter=args.niter)
if(args.preTrain):
	model.load_state_dict(torch.load(args.model))
	print("Loaded pre-trained model...")
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)


def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(epoch, batch, val_batch):
    t = time.time()
    sequence = np.random.permutation(len(batch))

    loss_all = []
    acc_all = []
    dice_all = []
    num_batch = int(len(batch) / mini)

    for b in range(num_batch):
        model.train()
        optimizer.zero_grad()

        for i in sequence[b * mini: (b + 1) * mini]:

            # Load sample and take one optimizer step
            # This is what is slowing down: reading
            # from a dict. Could not do
            # anything better here, as yet!
            idx_all = batch[i]['idx_all']
            x_idx_all = batch[i]['x_idx_all']
            y_idx_all = batch[i]['y_idx_all']
            num_nodes = batch[i]['num_nodes']
            adj = batch[i]['adj']
            adj_flat = batch[i]['adj_flat']
            features = batch[i]['features']

            if args.cuda:
                model.cuda(args.gpu)

            n2e_in = batch[i]['n2e_in']
            n2e_out = batch[i]['n2e_out']

            features, adj = Variable(features), Variable(adj)
            n2e_in, n2e_out = Variable(n2e_in), Variable(n2e_out)
            adj_flat = Variable(adj_flat)

            preds, embeddings = model(features, n2e_in, n2e_out, x_idx_all,
                                      y_idx_all)

            loss_train = (dice_loss(preds, adj_flat) + F.binary_cross_entropy(preds,adj_flat))/2
            loss_all.append(loss_train.item())

            acc_train = binary_accuracy(preds, adj_flat)
            acc_all.append(acc_train.item())

            dice_train = dice(preds, adj_flat)
            dice_all.append(dice_train.item())

            loss_train.backward()
        optimizer.step()
#    pdb.set_trace()
    loss_val, loss_std, dice_val, dice_std = evaluate(val_batch, save=False)

    writeLog(logFile, epoch, np.mean(loss_all), np.std(loss_all), np.mean(dice_all), np.std(dice_all),
                          loss_val, loss_std, dice_val, dice_std, time.time()-t)

    return np.mean(loss_all), np.mean(dice_all), np.mean(loss_val), np.mean(dice_val)


def evaluate(batch, save=False):
    loss_all = []
    acc_all = []
    dice_all = []
    for i in range(len(batch)):
        idx_all = batch[i]['idx_all']
        x_idx_all = batch[i]['x_idx_all']
        y_idx_all = batch[i]['y_idx_all']
        num_nodes = batch[i]['num_nodes']
        adj = batch[i]['adj']
        adj_flat = batch[i]['adj_flat']
        features = batch[i]['features']

        if args.cuda:
            model.cuda(args.gpu)

        features, adj = Variable(features), Variable(adj)
        adj_flat = Variable(adj_flat)

        model.eval()

        n2e_in = batch[i]['n2e_in']
        n2e_out = batch[i]['n2e_out']

        n2e_in, n2e_out = Variable(n2e_in), Variable(n2e_out)

        preds, embeddings = model(features, n2e_in, n2e_out, x_idx_all,
                                  y_idx_all)

        loss = (dice_loss(preds, adj_flat) + F.binary_cross_entropy(preds,adj_flat))/2

        loss_all.append(loss.item())

        acc = binary_accuracy(preds, adj_flat)
        acc_all.append(acc.item())

        dice_val = dice(preds, adj_flat)
        dice_all.append(dice_val.item())

        if save:
            adj_pred = np.array(preds.cpu().data.numpy(), dtype=np.float16)
            adj_pred = sp.csr_matrix((adj_pred, (x_idx_all.cpu().numpy(),
                                                 y_idx_all.cpu().numpy())),
                                     shape=(num_nodes, num_nodes))
            print('dice_test:{:.4f}'.format(dice_val.item()))
            np.savez_compressed(batch[i]['vol_id'] + '_emb.npz', np.array(embeddings.cpu().data.numpy(),dtype=np.float16))
    
            np.savez_compressed(batch[i]['vol_id'] + '.npz', np.array(adj_pred.todense()))
            print("Done! Predicted graph saved to " + batch[i]['vol_id'] + ".npz'")

    return np.mean(loss_all), np.std(loss_all), np.mean(dice_all), np.std(dice_all)


# Train model
t_total = time.time()
loss_tr = np.zeros(args.epochs)
loss_vl = np.zeros(args.epochs)
dice_tr = np.zeros(args.epochs)
dice_vl = np.zeros(args.epochs)

for epoch in range(args.epochs):
    loss_tr[epoch], dice_tr[epoch], loss_vl[epoch], dice_vl[epoch] = train(epoch, train_data, val_data)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
np.savez_compressed('loss_tr.npz', loss_tr)
np.savez_compressed('loss_vl.npz', loss_vl)
np.savez_compressed('dice_tr.npz', dice_tr)
np.savez_compressed('dice_vl.npz', dice_vl)

if(args.save):
	cTime = time.strftime("%Y_%m_%d_%H_%M")
	torch.save(model.state_dict(), cTime+'_hid'+repr(args.hidden)+'_module.pkl')
	print("Saved trained model")

print("Calculating full prediction on validation set...")

# Test : just pass the index of the image in batch
evaluate(val_data, save=True)

