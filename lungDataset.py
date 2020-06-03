import glob
from utils import *
import gzip
from pickle import load
from torch.utils.data import Dataset, DataLoader
import time
from os.path import splitext, basename

class lungDataset(Dataset):
    """Lung Graph dataset. Moved Thomas' preprocessing bits here as well"""

    def __init__(self, root_dir, cuda=False, gpu=0):
        """
        Args:
            root_dir (string): Directory with all the .pgz files
            cuda,gpu: Move dataset to GPU
        """
        self.root_dir = root_dir
        self.cuda = cuda
        self.gpu = gpu

    def __len__(self):
        return len(glob.glob(self.root_dir + '/*.npz'))

    def __getitem__(self, idx):
        # Unpacking is major bottleneck here
        img_name = sorted(glob.glob(self.root_dir + '/*.npz'))[idx]
#        nodes, adj_dense, adj_target = unpack(img_name)
        nodes, adj_dense, adj_target = unpackNpz(img_name)

        features = np.array(np.transpose(nodes), dtype=np.float32)

        adj = sp.csr_matrix(adj_dense, dtype=np.float32)

        # Node to edge transformer matrices
        n2e_in = sp.csr_matrix((np.ones(adj.nnz),
                                (np.arange(adj.nnz), sp.find(adj)[1])),
                               shape=(adj.nnz, adj.shape[0]))
        n2e_out = sp.csr_matrix((np.ones(adj.nnz),
                                 (np.arange(adj.nnz), sp.find(adj)[0])),
                                shape=(adj.nnz, adj.shape[0]))

        # Normalize
        features = (features - features.min(0)) * 2 / \
                   (features.max(0) - features.min(0)) - 1
        adj_dense = row_normalize(adj_dense)

        adj = sp.csr_matrix(adj_dense, dtype=np.float32)
        adj_target = sp.csr_matrix(adj_target, dtype=np.float32)

        num_nodes = adj.shape[0]

        idx_pos = to_linear_idx(sp.find(adj_target)[0],
                                sp.find(adj_target)[1],
                                num_nodes)
        idx_neg = to_linear_idx(sp.find(adj)[0], sp.find(adj)[1], num_nodes)

        idx_neg = np.setdiff1d(idx_neg, idx_pos)

        features = torch.FloatTensor(features)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        n2e_in = sparse_mx_to_torch_sparse_tensor(n2e_in)
        n2e_out = sparse_mx_to_torch_sparse_tensor(n2e_out)

        adj_flat = np.array(adj_target.todense(), dtype=np.float32).reshape(-1)
        adj_flat = torch.FloatTensor(adj_flat)

        idx_all = np.hstack((idx_pos, idx_neg))
        idx_all = np.array(idx_all, dtype=np.int64)
        x_idx_all, y_idx_all = to_2d_idx(idx_all, num_nodes)

        idx_all = torch.LongTensor(idx_all)
        x_idx_all = torch.LongTensor(x_idx_all)
        y_idx_all = torch.LongTensor(y_idx_all)

        adj_flat = adj_flat[idx_all]

        if (self.cuda):
            features = features.cuda(self.gpu)
            adj = adj.cuda(self.gpu)
            adj_flat = adj_flat.cuda(self.gpu)
            idx_all = idx_all.cuda(self.gpu)
            x_idx_all = x_idx_all.cuda(self.gpu)
            y_idx_all = y_idx_all.cuda(self.gpu)
            n2e_in = n2e_in.cuda(self.gpu)
            n2e_out = n2e_out.cuda(self.gpu)

        sample = {'features': features, 'adj': adj, 'adj_flat': adj_flat,
                  'num_nodes': num_nodes, 'idx_all': idx_all,
                  'x_idx_all': x_idx_all, 'y_idx_all': y_idx_all,
                  'idx_neg': idx_neg, 'idx_pos': idx_pos,
                  'n2e_in': n2e_in, 'n2e_out': n2e_out, 
		  'vol_id': splitext(basename(img_name))[0]}

        return sample
