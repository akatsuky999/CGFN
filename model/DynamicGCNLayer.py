import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DynamicGCNLayer_A(nn.Module):
    """
    \mathbf{A}=ReLU(\mathbf{W})
    """

    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(num_nodes, num_nodes))
        self.linear = nn.Linear(in_channels, out_channels)

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.linear.weight)

    def normalize_dynamic_adj(self, adj):
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0).repeat(adj.size(0), 1, 1)
        adj = adj + eye
        rowsum = adj.sum(dim=2).clamp(min=1e-12)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        return adj * d_inv_sqrt.unsqueeze(2) * d_inv_sqrt.unsqueeze(1)

    def forward(self, x):
        B, N, _ = x.shape
        dynamic_adj = F.relu(self.W).unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
        dynamic_adj = self.normalize_dynamic_adj(dynamic_adj)
        transformed_x = self.linear(x)
        output = torch.bmm(dynamic_adj, transformed_x)
        return output, dynamic_adj


class DynamicGCNLayer_Att(nn.Module):
    """
    \mathbf{A}=softmax(\frac{(XW_1)(XW_2)^T}{\sqrt{d}})
    """

    def __init__(self, in_channels, out_channels, num_nodes, temperature=None):
        super().__init__()
        self.W1 = nn.Linear(in_channels, out_channels)
        self.W2 = nn.Linear(in_channels, out_channels)
        self.linear = nn.Linear(in_channels, out_channels)

        self.temperature = temperature if temperature is not None else out_channels ** 0.5

        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def normalize_dynamic_adj(self, adj):
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0).repeat(adj.size(0), 1, 1)
        adj = adj + eye
        rowsum = adj.sum(dim=2).clamp(min=1e-12)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        return adj * d_inv_sqrt.unsqueeze(2) * d_inv_sqrt.unsqueeze(1)

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.W1(x)  # [B, N, out_channels]
        K = self.W2(x)  # [B, N, out_channels]
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # [B, N, N]
        attention_scores = attention_scores / self.temperature
        dynamic_adj = F.softmax(attention_scores, dim=-1)
        dynamic_adj = self.normalize_dynamic_adj(dynamic_adj)
        transformed_x = self.linear(x)
        output = torch.bmm(dynamic_adj, transformed_x)

        return output, dynamic_adj


class DynamicGCNLayer_DA(nn.Module):
    """
    \mathbf{DA}=ReLU\left(tanh\left(\alpha(\mathrm{DE}_{1}\mathrm{DE}_{2}^{T}
            -\mathrm{DE}_{2}\mathrm{DE}_{1}^{T})\right)\right)
    """

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


class DynamicGCNLayer_EE(nn.Module):
    """
    \mathbf{A}=ReLU(\mathrm{tanh}(\alpha(\mathbb{E}_1\mathbb{E}_2^T))
    """

    def __init__(self, in_channels, out_channels, num_nodes, alpha=1.0):
        super().__init__()
        self.alpha = alpha
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
        transformed_x = self.linear(x)
        node_indices = torch.arange(N, device=x.device)
        E1_emb = self.E1(node_indices).unsqueeze(0).expand(B, -1, -1)  # [B, N, out_channels]
        E2_emb = self.E2(node_indices).unsqueeze(0).expand(B, -1, -1)  # [B, N, out_channels]
        #  A = ReLU(tanh(α(E1E2^T)))
        cross_term = torch.bmm(E1_emb, E2_emb.transpose(1, 2))  # [B, N, N]
        dynamic_adj = F.relu(torch.tanh(self.alpha * cross_term))
        dynamic_adj = self.normalize_dynamic_adj(dynamic_adj)
        output = torch.bmm(dynamic_adj, transformed_x)
        return output, dynamic_adj


class DynamicGCNLayer(nn.Module):
    """
    \mathbf{A}^{\prime}=\gamma\odot(\tilde{\mathbf{H}}_{sym}\tilde{\mathbf{H}}_{sym}^{T})
            +(1-\gamma)\odot(\mathbf{H}_{asym}^{+}(\mathbf{H}_{asym}^{-})^{T})
    """

    def __init__(self, in_channels, out_channels, num_nodes,
                 sym_ratio=0.5, alpha_pre=1.0):
        super().__init__()

        self.out_channels = out_channels
        self.sym_ratio = sym_ratio
        self.alpha_pre = alpha_pre
        self.E_sym = nn.Embedding(num_nodes, int(out_channels * sym_ratio))
        self.E_asym = nn.Embedding(num_nodes, out_channels - int(out_channels * sym_ratio))
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)

        self.gate = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        nn.init.xavier_uniform_(self.E_sym.weight)
        nn.init.xavier_uniform_(self.E_asym.weight)

    def normalize_dynamic_adj(self, adj):
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0).repeat(adj.size(0), 1, 1)
        adj = adj + eye
        rowsum = adj.sum(dim=2).clamp(min=1e-12)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        return adj * d_inv_sqrt.unsqueeze(2) * d_inv_sqrt.unsqueeze(1)

    def forward(self, x):
        B, N, _ = x.shape

        base = self.linear1(x)  # [B,N,d_out]

        node_indices = torch.arange(N, device=x.device)
        E_sym = self.E_sym(node_indices).unsqueeze(0).expand(B, -1, -1)  # [B,N,d_sym]
        E_asym = self.E_asym(node_indices).unsqueeze(0).expand(B, -1, -1)  # [B,N,d_asym]

        base_sym = base[..., :int(self.out_channels * self.sym_ratio)]
        base_asym = base[..., int(self.out_channels * self.sym_ratio):]

        H_sym = torch.tanh(self.alpha_pre * (base_sym + E_sym))
        A_sym = torch.bmm(H_sym, H_sym.transpose(1, 2))  # [B,N,N]

        H_asym_pos = torch.tanh(self.alpha_pre * (base_asym + E_asym))
        H_asym_neg = torch.tanh(self.alpha_pre * (base_asym - E_asym))
        A_asym = torch.bmm(H_asym_pos, H_asym_neg.transpose(1, 2))  # [B,N,N]

        gate_input = torch.cat([
            A_sym.unsqueeze(-1),
            A_asym.unsqueeze(-1)
        ], dim=-1).reshape(B, N * N, 2)  # [B,N*N,2]

        gamma = self.gate(gate_input).reshape(B, N, N)  # [B,N,N]
        DA = gamma * A_sym + (1 - gamma) * A_asym
        DA = F.relu(DA)
        DA = self.normalize_dynamic_adj(DA)

        output = torch.bmm(DA, x)  # [B,N,d_out]
        output = self.linear2(output)

        return output, DA


class DynamicGCNLayer_r(nn.Module):
    """
    \mathbf{A}^{\prime}=\gamma\odot(\tilde{\mathbf{H}}_{sym}\tilde{\mathbf{H}}_{sym}^{T})
            +(1-\gamma)\odot(\mathbf{H}_{asym}^{+}(\mathbf{H}_{asym}^{-})^{T})
    """

    def __init__(self, in_channels, out_channels, num_nodes,
                 sym_ratio=0.5, alpha_pre=1.0, rank=16):
        super().__init__()

        self.out_channels = out_channels
        self.sym_ratio = sym_ratio
        self.alpha_pre = alpha_pre
        self.rank = rank
        self.E_sym_U = nn.Embedding(num_nodes, rank)
        self.E_sym_V = nn.Parameter(torch.Tensor(rank, int(out_channels * sym_ratio)))
        self.E_asym_U = nn.Embedding(num_nodes, rank)
        self.E_asym_V = nn.Parameter(torch.Tensor(rank, out_channels - int(out_channels * sym_ratio)))

        nn.init.xavier_uniform_(self.E_sym_U.weight)
        nn.init.xavier_uniform_(self.E_sym_V)
        nn.init.xavier_uniform_(self.E_asym_U.weight)
        nn.init.xavier_uniform_(self.E_asym_V)

        self.linear = nn.Linear(in_channels, out_channels)

        self.gate = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def normalize_dynamic_adj(self, adj):
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0).repeat(adj.size(0), 1, 1)
        adj = adj + eye
        rowsum = adj.sum(dim=2).clamp(min=1e-12)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        return adj * d_inv_sqrt.unsqueeze(2) * d_inv_sqrt.unsqueeze(1)

    def compute_A_sym(self, base_sym, E_sym):
        H_sym = torch.tanh(self.alpha_pre * (base_sym + E_sym))
        return torch.bmm(H_sym, H_sym.transpose(1, 2))

    def compute_A_asym(self, base_asym, E_asym):
        H_asym_pos = torch.tanh(self.alpha_pre * (base_asym + E_asym))
        H_asym_neg = torch.tanh(self.alpha_pre * (base_asym - E_asym))
        return torch.bmm(H_asym_pos, H_asym_neg.transpose(1, 2))

    def forward(self, x):
        B, N, _ = x.shape
        base = self.linear(x)

        node_indices = torch.arange(N, device=x.device)
        E_sym = torch.mm(self.E_sym_U(node_indices), self.E_sym_V).unsqueeze(0).expand(B, -1, -1)
        E_asym = torch.mm(self.E_asym_U(node_indices), self.E_asym_V).unsqueeze(0).expand(B, -1, -1)
        split_point = int(self.out_channels * self.sym_ratio)
        base_sym, base_asym = base.split([split_point, self.out_channels - split_point], dim=-1)

        A_sym = checkpoint(self.compute_A_sym, base_sym, E_sym, use_reentrant=False)
        A_asym = checkpoint(self.compute_A_asym, base_asym, E_asym, use_reentrant=False)
        gate_input = torch.stack([A_sym, A_asym], dim=-1).view(B, N * N, 2)
        gamma = self.gate(gate_input).view(B, N, N)

        DA = gamma * A_sym + (1 - gamma) * A_asym
        DA = F.relu(DA, inplace=True)  # In-place
        DA = self.normalize_dynamic_adj(DA)

        del A_sym, A_asym, gamma, gate_input, E_sym, E_asym, base_sym, base_asym
        torch.cuda.empty_cache()

        return torch.bmm(DA, base), DA
