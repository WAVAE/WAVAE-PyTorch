import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import rnn


class AddEps(nn.Module):
    def __init__(self, channels):
        super(AddEps, self).__init__()

        self.channels = channels
        self.linear = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh()
        )
    def forward(self, x):
        eps = torch.randn_like(x)
        eps = self.linear(eps)
        return eps + x
class Encoder(nn.Module):
    def __init__(self, hyper_params, in_shape, out_shape):
        super(Encoder, self).__init__()

        self.hyper_params = hyper_params
        self.dot_cnn1 = nn.Sequential(
            nn.Conv1d(in_shape, in_shape, kernel_size=1, stride=1),
            nn.Softplus()
        )
        self.dot_cnn2 = nn.Sequential(
            nn.Conv1d(2 * in_shape, in_shape, kernel_size=1, stride=1),
            nn.Softplus()
        )
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_shape, 2*in_shape,
                      kernel_size=5, stride=1, padding=4),
            nn.Softplus()
        )

        self.eps = AddEps(in_shape)
        self.linear_o = nn.Linear(in_shape, out_shape)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):
        if self.hyper_params['add_eps']:
            y = self.dot_cnn1(self.eps(x).transpose(1, 2)).transpose(1, 2)
            y = self.cnn_layer(self.eps(y).transpose(1, 2))[:, :, :-4]
        else:
            y = self.dot_cnn1(x.transpose(1, 2)).transpose(1, 2)
            y = self.cnn_layer(y.transpose(1, 2))[:, :, :-4]
        y = self.dot_cnn2(y).transpose(1, 2)
        y = x + y

        y = self.dropout(y)
        out = self.linear_o(y)

        return out

class Decoder(nn.Module):
    def __init__(self, hyper_params):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(
            hyper_params['latent_size'], hyper_params['item_embed_size'])
        self.hyper_params=hyper_params
        self.linear2 = nn.Linear(
            hyper_params['item_embed_size'], hyper_params['total_items'] + 1)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        out_embed = x
        x = self.activation(x)
        x = self.linear2(x)
        return x, out_embed


class Model(nn.Module):
    def __init__(self, hyper_params):
        super(Model, self).__init__()
        self.hyper_params = hyper_params
        self.decoder = Decoder(hyper_params)
        self.item_embed = nn.Embedding(
            hyper_params['total_items'] + 1, hyper_params['item_embed_size'])
        self.gru = nn.GRU(
            hyper_params['item_embed_size']  +hyper_params['total_items'] + 1, hyper_params['rnn_size'],
            batch_first=True, num_layers=1
        )
        self.linear_o = nn.Linear(
            hyper_params['hidden_size'], hyper_params['latent_size'])
        self.linear1 = nn.Linear(
            hyper_params['hidden_size'], 2 * hyper_params['latent_size'])
        nn.init.xavier_normal_(self.linear1.weight)

        self.tanh = nn.Tanh()
        self.embed_dropout = nn.Dropout(0.5)
        self.encoder=Encoder(hyper_params,
                                        hyper_params['rnn_size'], hyper_params['latent_size'])
    def sample_latent(self, z_inferred):
        return torch.randn_like(z_inferred)

    def forward(self,x):
        x1=F.one_hot(x, num_classes=self.hyper_params['total_items']+1)
        x = self.item_embed(x)
        x_real = x
        x =torch.cat([x,x1],dim=-1)
        rnn_out, _ = self.gru(self.embed_dropout(x))
        z_inferred = self.encoder(rnn_out)
        dec_out, out_embed = self.decoder(z_inferred)
        return dec_out, x_real,z_inferred, out_embed

class Embed(nn.Module):
    def __init__(self, hyper_params):
        super(Embed, self).__init__()

        self.item_embed = nn.Embedding(
            hyper_params['total_items'] + 1, hyper_params['item_embed_size'])

    def forward(self, x):
        return self.item_embed(x)
class Adversary(nn.Module):
    def __init__(self, hyper_params):
        super(Adversary, self).__init__()
        self.hyper_params = hyper_params
        self.linear_i = nn.Linear(
            hyper_params['item_embed_size'] + hyper_params['latent_size'], 128)

        self.dnet_list = []
        self.net_list = []
        for _ in range(2):
            self.dnet_list.append(nn.Linear(128, 128))
            self.net_list.append(nn.Linear(128, 128))

        self.dnet_list = nn.ModuleList(self.dnet_list)
        self.net_list = nn.ModuleList(self.net_list)

        self.linear_o = nn.Linear(128, hyper_params['latent_size'])
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x, z, padding):
        # batch_size x seq_len x dim
        net = torch.cat((x, z), 2) #将x和z按列拼接
        net = self.linear_i(net)
        net = self.dropout1(net)

        for i in range(2):
            dnet = self.dnet_list[i](net)
            net = net + self.net_list[i](dnet)
            net = F.elu(net)

        # seq_len
        net = self.linear_o(net)
        net = self.dropout2(net)
        net = net + 0.5 * torch.square(z)
        net = net * (1.0 - padding.float().unsqueeze(2))

        return net
class D(nn.Module):
    def __init__(self,hyper_params):
        super(D, self).__init__()
        self.linear1=nn.Linear(hyper_params['total_items']+1, 128)
        self.relu=nn.ReLU()
        self.liner2=nn.Linear(128, 64)
        self.sigmode=nn.Sigmoid()
    def forward(self,x):
        x1=x.float()
        x=self.linear1(x1)
        x=self.relu(x)
        x=self.liner2(x)
        #x=self.sigmode(x)
        return x