import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import eigs

# ---------------------- base ----------------------
def scaled_Laplacian(W):
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


# ---------------------- Att ----------------------
class Temporal_Attention_layer(nn.Module):
    def __init__(self, DEVICE, in_channels, num_of_vertices, windows):
        super().__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        self.be = nn.Parameter(torch.FloatTensor(1, windows, windows).to(DEVICE))
        self.Ve = nn.Parameter(torch.FloatTensor(windows, windows).to(DEVICE))

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        rhs = torch.matmul(self.U3, x)
        product = torch.matmul(lhs, rhs)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        return F.softmax(E, dim=1)


# ---------------------- DyGCN ----------------------
class StaticGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, static_adj):
        return self.linear(torch.bmm(static_adj, x))


class DynamicGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, alpha_pre=1.0, alpha_da=1.0):
        super().__init__()
        self.alpha_pre = alpha_pre
        self.alpha_da = alpha_da
        self.E1 = nn.Embedding(num_nodes, out_channels)
        self.E2 = nn.Embedding(num_nodes, out_channels)
        self.linear = nn.Linear(in_channels, out_channels)
        nn.init.xavier_uniform_(self.E1.weight)
        nn.init.xavier_uniform_(self.E2.weight)

    def normalize_dynamic_adj(self, adj):
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0).repeat(adj.size(0), 1, 1)
        adj = adj + eye
        rowsum = adj.sum(dim=2).clamp(min=1e-12)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        return adj * d_inv_sqrt.unsqueeze(2) * d_inv_sqrt.unsqueeze(1)

    def forward(self, x):
        B, N, _ = x.shape
        base = self.linear(x)
        node_indices = torch.arange(N, device=x.device)
        E1_emb = self.E1(node_indices).unsqueeze(0).expand(B, -1, -1)
        E2_emb = self.E2(node_indices).unsqueeze(0).expand(B, -1, -1)
        DE1 = torch.tanh(self.alpha_pre * (base * E1_emb))
        DE2 = torch.tanh(self.alpha_pre * (base * E2_emb))
        cross_term = torch.bmm(DE1, DE2.transpose(1, 2)) - torch.bmm(DE2, DE1.transpose(1, 2))
        DA = F.relu(torch.tanh(self.alpha_da * cross_term))
        DA = self.normalize_dynamic_adj(DA)
        return self.linear(torch.bmm(DA, x)), DA


class DyGCN(nn.Module):
    def __init__(self, in_channels, out_channels, static_adj, num_nodes, windows=12, lambda_=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.static_adj = static_adj
        self.linear = nn.Linear(in_channels, out_channels)
        self.static_gcn = nn.ModuleList([StaticGCNLayer(in_channels, out_channels) for _ in range(windows)])
        self.dynamic_gcn = nn.ModuleList([DynamicGCNLayer(in_channels, out_channels, num_nodes) for _ in range(windows)])

    def forward(self, x):
        B, N, C, T = x.shape
        outputs = []
        for t in range(T):
            x_t = x[..., t]
            x_0 = self.linear(x_t)
            static_adj = self.static_adj.unsqueeze(0).repeat(B, 1, 1)
            static_out = self.static_gcn[t](x_t, static_adj)
            dynamic_out, _ = self.dynamic_gcn[t](x_t)
            fused = (self.lambda_ - 0.025) * static_out + (1 - self.lambda_ - 0.025) * dynamic_out + 0.05 * x_0
            outputs.append(fused.unsqueeze(-1))
        return torch.cat(outputs, dim=-1)


# ---------------------- core ----------------------
class TAD_block(nn.Module):
    def __init__(self, DEVICE, in_channels, hidden_layer, hidden_time_layer,
                 time_strides, static_adj, num_of_vertices, windows):
        super().__init__()
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, windows)
        self.dygcn = DyGCN(in_channels=in_channels,
                           out_channels=hidden_layer,
                           static_adj=static_adj,
                           num_nodes=num_of_vertices,
                           windows=windows)
        self.time_conv = nn.Conv2d(hidden_layer, hidden_time_layer,
                                   kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, hidden_time_layer,
                                       kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(hidden_time_layer)

    def forward(self, x):
        B, N, C, T = x.shape
        temporal_At = self.TAt(x)
        x_TAt = torch.matmul(x.reshape(B, -1, T), temporal_At).reshape(B, N, C, T)
        spatial_gcn = self.dygcn(x_TAt)
        time_conv_out = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        residual = self.residual_conv(x.permute(0, 2, 1, 3))
        output = F.relu(residual + time_conv_out)
        return self.ln(output.permute(0, 3, 2, 1)).permute(0, 2, 3, 1)


class TAD_submodule(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, hidden_layer, hidden_time_layer,
                 time_strides, static_adj, num_for_predict, len_input, num_of_vertices, emb):
        super().__init__()
        self.device = DEVICE
        self.node_emb = nn.Embedding(num_of_vertices, emb)
        self.time_in_day_emb = nn.Embedding(288, emb)
        self.day_in_week_emb = nn.Embedding(7, emb)
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.time_in_day_emb.weight)
        nn.init.xavier_uniform_(self.day_in_week_emb.weight)

        L_tilde = scaled_Laplacian(static_adj.cpu().numpy())
        self.static_adj = torch.from_numpy(L_tilde).float().to(DEVICE)
        self.BlockList = nn.ModuleList([
            TAD_block(DEVICE, in_channels, hidden_layer, hidden_time_layer,
                          time_strides, self.static_adj, num_of_vertices, len_input)
        ])
        self.BlockList.extend([
            TAD_block(DEVICE, hidden_time_layer, hidden_layer, hidden_time_layer,
                          1, self.static_adj, num_of_vertices, len_input // time_strides)
            for _ in range(nb_block - 1)
        ])
        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict,
                                    kernel_size=(1, hidden_time_layer))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        B, T, N, C = x.shape

        time_idx = (x[:, -1, :, 1] * 287).long().clamp(0, 287)
        day_idx = (x[:, -1, :, 2] * 6).long().clamp(0, 6)
        node_idx = torch.arange(N).to(x.device)

        time_emb = self.time_in_day_emb(time_idx).unsqueeze(1).expand(-1, T, -1, -1)
        day_emb = self.day_in_week_emb(day_idx).unsqueeze(1).expand(-1, T, -1, -1)
        node_emb = self.node_emb(node_idx).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        x = torch.cat([
            x[..., 0:1],
            time_emb,
            day_emb,
            node_emb
        ], dim=-1)

        x = x.permute(0, 2, 3, 1)

        for block in self.BlockList:
            x = block(x)
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return output


# ---------------------- main ----------------------
def TAD_Net(DEVICE, nb_block, in_channels, hidden_layer, hidden_time_layer,
               time_strides, adj_mx, num_for_predict, len_input, num_of_vertices, emb):
    static_adj = torch.from_numpy(adj_mx).float().to(DEVICE)
    model = TAD_submodule(DEVICE, nb_block, in_channels, hidden_layer,
                              hidden_time_layer, time_strides, static_adj,
                              num_for_predict, len_input, num_of_vertices, emb)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model