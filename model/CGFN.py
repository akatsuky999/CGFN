import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import eigs
from torch.utils.checkpoint import checkpoint
from .DynamicGCNLayer import *


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
    def __init__(self, in_channels, out_channels, num_nodes, mask_type="sigmoid", init_strategy="random"):
        super().__init__()
        self.num_nodes = num_nodes
        self.mask_type = mask_type
        self.linear = nn.Linear(in_channels, out_channels)
        self.mask = nn.Parameter(torch.Tensor(num_nodes, num_nodes))

        if init_strategy == "ones":
            nn.init.ones_(self.mask)
        elif init_strategy == "random":
            nn.init.uniform_(self.mask, 0.8, 1.2)
        elif init_strategy == "eye":
            nn.init.eye_(self.mask)
        else:
            raise ValueError("Unknown init strategy")

        self.register_buffer("static_adj", None)

    def _apply_mask(self, adj):
        if self.mask_type == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_type == "softmax":
            mask = F.softmax(self.mask, dim=1)
        elif self.mask_type == "tanh":
            mask = (torch.tanh(self.mask) + 1) / 2
        else:
            raise NotImplementedError

        return adj * mask.unsqueeze(0)

    def forward(self, x, static_adj):
        if self.static_adj is None:
            self.static_adj = static_adj.detach()

        enhanced_adj = self._apply_mask(static_adj)
        x = torch.bmm(enhanced_adj, x)
        return self.linear(x)



class DyGCN(nn.Module):
    def __init__(self, in_channels, out_channels, static_adj, num_nodes, windows=12, lambda_=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.static_adj = static_adj
        self.linear = nn.Linear(in_channels, out_channels)
        self.static_gcn = nn.ModuleList([StaticGCNLayer(in_channels, out_channels, num_nodes) for _ in range(windows)])
        self.dynamic_gcn = nn.ModuleList(
            [DynamicGCNLayer(in_channels, out_channels, num_nodes) for _ in range(windows)])

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
class GroupDualGTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size, groups=4,
                 act_p_type='mish', act_q_type='hardtanh'):
        super().__init__()
        assert in_channels % groups == 0, "The number of channels must be divisible by the number of groups."

        self.conv = nn.Conv2d(
            in_channels,
            2 * in_channels,
            kernel_size=(1, kernel_size),
            stride=(1, time_strides),
            groups=groups,
            padding=0)

        self.act_p = self._select_activation(act_p_type)
        self.act_q = self._select_activation(act_q_type, is_gate=True)
        self.gn_p = nn.GroupNorm(groups, in_channels)
        self.gn_q = nn.GroupNorm(groups, in_channels)

    def _select_activation(self, act_type, is_gate=False):
        if is_gate:
            return {
                'hardtanh': nn.Hardtanh(min_val=0.0, max_val=1.0),
                'sigmoid': nn.Sigmoid(),
                'swish': nn.SiLU()
            }[act_type.lower()]
        else:
            return {
                'mish': nn.Mish(),
                'tanh': nn.Tanh(),
                'gelu': nn.GELU()
            }[act_type.lower()]

    def forward(self, x):
        x_conv = self.conv(x)
        x_p, x_q = x_conv.chunk(2, dim=1)
        x_p = self.gn_p(x_p)
        x_q = self.gn_q(x_q)

        return torch.mul(self.act_p(x_p), self.act_q(x_q))


class CGFN_block(nn.Module):
    def __init__(self, DEVICE, in_channels, hidden_layer, hidden_time_layer,
                 time_strides, static_adj, num_of_vertices, windows):
        super().__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.TAt = Temporal_Attention_layer(DEVICE, in_channels, num_of_vertices, windows)
        self.dygcn = DyGCN(in_channels=in_channels,
                           out_channels=hidden_layer,
                           static_adj=static_adj,
                           num_nodes=num_of_vertices,
                           windows=windows)
        # 定义不同卷积核大小的 GTU 层
        self.gtu3 = GroupDualGTU(hidden_time_layer, time_strides, 3)
        self.gtu5 = GroupDualGTU(hidden_time_layer, time_strides, 5)
        self.gtu7 = GroupDualGTU(hidden_time_layer, time_strides, 7)
        self.gtu9 = GroupDualGTU(hidden_time_layer, time_strides, 9)

        self.fcmy = nn.Sequential(
            nn.Linear(4 * windows - 20, windows),
            nn.Dropout(0.05),
        )
        self.time_conv = nn.Conv2d(hidden_layer, hidden_time_layer,
                                   kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, hidden_time_layer,
                                       kernel_size=(1, 1), stride=(1, 1))
        self.ln = nn.LayerNorm(hidden_time_layer)

    def forward(self, x):
        B, N, C, T = x.shape
        temporal_At = self.TAt(x)
        x_TAt = torch.matmul(x.reshape(B, -1, T), temporal_At).reshape(B, N, C, T)
        spatial_gcn = self.dygcn(x_TAt)
        X = spatial_gcn.permute(0, 2, 1, 3)  # B,F,N,T
        x_gtu = []
        x_gtu.append(self.gtu3(X))  # B,F,N,T-2
        x_gtu.append(self.gtu5(X))  # B,F,N,T-4
        x_gtu.append(self.gtu7(X))  # B,F,N,T-6
        x_gtu.append(self.gtu9(X))  # B,F,N,T-8
        time_conv = torch.cat(x_gtu, dim=-1)  # B,F,N, 4T-20
        time_conv = self.fcmy(time_conv)
        time_conv_output = self.relu(time_conv)
        residual = self.residual_conv(x.permute(0, 2, 1, 3))  # B,F,N,T
        x_residual = self.ln(F.relu(residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return x_residual


class CGFN_MAIN(nn.Module):
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
            CGFN_block(DEVICE, in_channels, hidden_layer, hidden_time_layer,
                      time_strides, self.static_adj, num_of_vertices, len_input)
        ])
        self.BlockList.extend([
            CGFN_block(DEVICE, hidden_time_layer, hidden_layer, hidden_time_layer,
                      1, self.static_adj, num_of_vertices, len_input // time_strides)
            for _ in range(nb_block - 1)
        ])
        self.final_conv = nn.Conv2d(int((len_input * nb_block) / time_strides), num_for_predict,
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

        concats = []
        for block in self.BlockList:
            x = block(x)
            concats.append(x)

        final_x = torch.cat(concats, dim=-1)

        output = self.final_conv(final_x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return output


# ---------------------- main ----------------------
def CGFN(DEVICE, nb_block, in_channels, hidden_layer, hidden_time_layer,
            time_strides, adj_mx, num_for_predict, len_input, num_of_vertices, emb):
    static_adj = torch.from_numpy(adj_mx).float().to(DEVICE)
    model = CGFN_MAIN(DEVICE, nb_block, in_channels, hidden_layer,
                          hidden_time_layer, time_strides, static_adj,
                          num_for_predict, len_input, num_of_vertices, emb)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model