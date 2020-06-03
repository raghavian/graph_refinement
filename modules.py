import torch
import torch.nn as nn
import torch.nn.functional as F
import math

SIGMA = 1
EPSILON = 1e-5


class EdgeGNN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, niter):
        super(EdgeGNN, self).__init__()

        self.fc_node_1_1 = nn.Linear(nfeat, 2*nhid)
        self.fc_node_1_2 = nn.Linear(nhid, nhid)

        self.fc_edge_1_1 = nn.Linear(nhid * 2, nhid)
        self.fc_edge_1_2 = nn.Linear(nhid, nhid)

        self.fc_node_2_1 = nn.Linear(nhid * 2, nhid)
        self.fc_node_2_2 = nn.Linear(nhid, nhid)

        self.fc_edge_2_1 = nn.Linear(nhid * 2, nhid)
        self.fc_edge_2_2 = nn.Linear(nhid, nhid)

        self.fc_node_3_1 = nn.Linear(nhid * 2, nhid)
        self.fc_node_3_2 = nn.Linear(nhid, nhid)

        self.fc_edge_3_1 = nn.Linear(nhid * 2, nhid)
        self.fc_edge_3_2 = nn.Linear(nhid, nhid)

        self.fc_node_4_1 = nn.Linear(nhid * 2, nhid)
        self.fc_node_4_2 = nn.Linear(nhid, nhid)


        self.ln1 = LayerNorm(nhid)
        self.ln2 = LayerNorm(nhid)
        self.ln3 = LayerNorm(nhid)
        self.ln4 = LayerNorm(nhid)
        self.ln5 = LayerNorm(nhid)
        self.ln6 = LayerNorm(nhid)
        self.dec = nn.Linear(nhid,1)
        self.dropout = dropout
        self.niter = niter

    def encode(self, x, n2e_in, n2e_out):
        x = F.relu(self.fc_node_1_1(x))


        for _ in range(self.niter):
                # Node MLP
                x = F.relu(self.fc_node_2_1(x))
                x = F.dropout(x, self.dropout, training=self.training)
                x = F.relu(self.fc_node_2_2(x))

                # Node to edge
                x_in = SparseMM()(n2e_in, x)
                x_out = SparseMM()(n2e_out, x)
                x = torch.cat([x_in, x_out], 1)

                # Edge MLP
                x = F.relu(self.fc_edge_1_1(x))
                x = F.dropout(x, self.dropout, training=self.training)
                x = F.relu(self.fc_edge_1_2(x))
                xEdge = x

                # Edge to node
                x_in = SparseMM()(n2e_in.transpose(0, 1), x)
                x_out = SparseMM()(n2e_out.transpose(0, 1), x)
                x = torch.cat([x_in, x_out], 1)

        return xEdge

    def forward(self, inputs, n2e_in, n2e_out, x_idx, y_idx):
        z = self.encode(inputs, n2e_in, n2e_out)
        preds = torch.sigmoid(self.dec(z))
	#preds = F.sigmoid(self.dec(z.unsqueeze(2)))
        return preds.view(-1), z


class GAE(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GAE, self).__init__()

        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, nhid)
        self.fc4 = nn.Linear(nhid, nhid)

        self.gc1 = GraphConvolutionFirstOrder(nfeat, nhid)
        self.gc2 = GraphConvolutionFirstOrder(nhid, nhid)
        self.gc3 = GraphConvolutionFirstOrder(nhid, nhid)

        self.ln1 = LayerNorm(nhid)
        self.ln2 = LayerNorm(nhid)

        # self.dec = RadialBasisFunctionKernel()
        # self.dec = InnerProductDecoder()
        self.dec = MLPDecoder(nhid)
        self.dropout = dropout

    def encode(self, x, adj):
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)

        return x

    def decode(self, z, x_idx, y_idx):
        return self.dec(z, x_idx, y_idx)

    def forward(self, inputs, adj, x_idx, y_idx):
        z = self.encode(inputs, adj)
        return self.decode(z, x_idx, y_idx), z


class GAERecurrent(nn.Module):
    def __init__(self, nfeat, nhid, dropout, steps=5, gpu=0):
        super(GAERecurrent, self).__init__()

        self.fc1 = nn.Linear(nhid, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, nhid)

        self.hidden_r = GraphConvolutionFirstOrder(nhid, nhid, bias=False)
        self.hidden_i = GraphConvolutionFirstOrder(nhid, nhid, bias=False)
        self.hidden_h = GraphConvolutionFirstOrder(nhid, nhid, bias=False)

        self.input_r = nn.Linear(nfeat, nhid, bias=True)
        self.input_i = nn.Linear(nfeat, nhid, bias=True)
        self.input_n = nn.Linear(nfeat, nhid, bias=True)

        self.ln2 = LayerNorm(nhid)

        self.nhid = nhid

        # self.dec = RadialBasisFunctionKernel()

        self.dec = MLPDecoder(nhid)

        self.dropout = dropout

        self.num_recurrent_steps = steps

        self.gpu = gpu

    def encode(self, inputs, adj):

        # Initialize hidden state
        hidden = torch.autograd.Variable(
            torch.zeros(inputs.size(0), self.nhid))
        if inputs.is_cuda:
            hidden = hidden.cuda(self.gpu)

            # GRU-style gated aggregation
        for _ in range(self.num_recurrent_steps):
            r = F.sigmoid(self.input_r(inputs) + self.hidden_r(hidden, adj))
            i = F.sigmoid(self.input_i(inputs) + self.hidden_i(hidden, adj))
            n = F.tanh(self.input_n(inputs) + r * self.hidden_h(hidden, adj))
            hidden = (1 - i) * n + i * hidden

        # Output MLP
        x = F.relu(self.fc1(hidden))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc3(x))
        x = self.ln2(x)

        return x

    def decode(self, z, x_idx, y_idx):
        return self.dec(z, x_idx, y_idx)

    def forward(self, inputs, adj, x_idx, y_idx):
        z = self.encode(inputs, adj)
        return self.decode(z, x_idx, y_idx), z


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def forward(self, matrix1, matrix2):
        self.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    def backward(self, grad_output):
        matrix1, matrix2 = self.saved_tensors
        grad_matrix1 = grad_matrix2 = None

        if self.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if self.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = SparseMM()(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphConvolutionFirstOrder(nn.Module):
    """
    Simple GCN layer, with separate processing of self-connection
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionFirstOrder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neighbor = nn.Parameter(
            torch.Tensor(in_features, out_features))
        self.weight_self = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_neighbor.size(1))
        self.weight_neighbor.data.uniform_(-stdv, stdv)
        self.weight_self.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        act_self = torch.mm(input, self.weight_self)
        support_neighbor = torch.mm(input, self.weight_neighbor)
        act_neighbor = SparseMM()(adj, support_neighbor)
        output = act_self + act_neighbor
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class RadialBasisFunctionKernel(nn.Module):
    def __init__(self):
        super(RadialBasisFunctionKernel, self).__init__()

    def forward(self, inputs, x_idx, y_idx):
        x = inputs[x_idx]
        y = inputs[y_idx]
        dist = (x - y).pow(2).sum(1)
        se_dist = torch.exp(- dist / (2 * (SIGMA ** 2)))
        return (se_dist - EPSILON) * (1 - EPSILON) + EPSILON


class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, inputs, x_idx, y_idx):
        x = inputs[x_idx]
        y = inputs[y_idx]
        prod = (x * y).sum(1)
        return F.sigmoid(prod)


class MLPDecoder(nn.Module):
    def __init__(self, nhid):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(nhid * 2, nhid)
        self.fc2 = nn.Linear(nhid, 1)

    def forward(self, inputs, x_idx, y_idx):
        x = inputs[x_idx]
        y = inputs[y_idx]
        h = torch.cat((x, y), 1)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        print(h.size())
        return torch.sigmoid(h)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
