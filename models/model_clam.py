import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""

class ExpertMLP(nn.Module):
    """
    简单的三层 MLP 专家：
        in_dim -> hidden_dim -> in_dim -> ReLU
    图中对应: L1, L2, L3, ReLU，平进平出
    """
    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * in_dim   # 默认中间层扩一倍

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
            nn.ReLU(),               # 图中最后的 ReLU
        )

    def forward(self, x):
        return self.net(x)

class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class CLAM_MB(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        
        self.proj_virchow = nn.Linear(2560, self.embed_dim)
    def forward(self, h_V, h_U, clinical,label=None, instance_eval=False, return_features=False, attention_only=False):

        
        
        
        h_V = self.proj_virchow(h_V)  # [N1, embed_dim]

        h = h_V

        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    
class CLAM_MB_ClinicalAdd(CLAM_SB):
    def __init__(self, 
                 gate=True, 
                 size_arg="small", 
                 dropout=0., 
                 k_sample=8, 
                 n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), 
                 subtyping=False, 
                 embed_dim=2560,
                 clinical_dim=6   # 临床向量维度
                 ):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]  # size[1] 是 bag 特征维度

        # --------- 原始的特征 + attention 部分不变 ----------
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)

        # bag-level classifiers（每个类一个 linear）
        # ⭐ 这里输入维度 = bag_feat_dim + clinical_dim
        self.bag_feat_dim = size[1]
        self.clinical_dim = clinical_dim
        self.classifiers = nn.ModuleList([
            nn.Linear(self.bag_feat_dim + self.clinical_dim, 1)
            for _ in range(n_classes)
        ])

        # instance-level classifiers（还是只用 WSI 特征）
        self.instance_classifiers = nn.ModuleList([nn.Linear(size[1], 2) for _ in range(n_classes)])

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        # ⭐ 纯 concat 版本其实可以不要 clinical_fc，如果你觉得需要再加一层也行
        # 这里先保留一个可选的小 MLP（如果想用可以打开）
        self.use_clinical_mlp = False
        if self.use_clinical_mlp:
            self.clinical_fc = nn.Sequential(
                nn.Linear(clinical_dim, clinical_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

    def forward(self, 
                h1, h,
                clinical,
                label=None, 
                instance_eval=False, 
                return_features=False, 
                attention_only=False
                ):
        # h: [N, embed_dim]
        A, h = self.attention_net(h)   # A: [N, K], h: [N, bag_feat_dim]
        A = torch.transpose(A, 1, 0)   # [K, N]

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)       # softmax over N

        # ------------------- instance_eval 部分保持原样 -------------------
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []

            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # [K]

            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]

                if inst_label == 1:  # in-the-class
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue

                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        # --------- bag-level pooled feature（视觉） ----------
        # M: [K, bag_feat_dim]，每一行是一个“类通道”的 bag 特征
        M = torch.mm(A, h)

        # --------- ⭐ 在这里与临床特征 concat ----------
        if clinical is not None:
            if clinical.dim() == 1:
                clinical = clinical.unsqueeze(0)  # [1, clinical_dim]
            clinical = clinical.to(M.device)

            # 可选：如果你打开了 self.use_clinical_mlp
            if hasattr(self, "use_clinical_mlp") and self.use_clinical_mlp:
                clinical = self.clinical_fc(clinical)  # [1, clinical_dim]

            # 重复到每个类通道： [K, clinical_dim]
            clin_rep = clinical.repeat(self.n_classes, 1)

            # 拼接： [K, bag_feat_dim + clinical_dim]
            M_fused = torch.cat([M, clin_rep], dim=1)
        else:
            M_fused = M

        # --------- bag-level logits 计算 ----------
        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M_fused[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        # instance_eval 输出
        if instance_eval:
            results_dict = {
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds)
            }
        else:
            results_dict = {}

        if return_features:
            results_dict.update({'features': M_fused})

        return logits, Y_prob, Y_hat, A_raw, results_dict

            

        # --------- bag-level logits 计算 ----------
        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M_fused[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        # instance_eval 输出
        if instance_eval:
            results_dict = {
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds)
            }
        else:
            results_dict = {}

        if return_features:
            # 这里既可以返回 M，也可以返回融合后的 M_fused，看你后续可视化需求
            results_dict.update({'features': M_fused})

        return logits, Y_prob, Y_hat, A_raw, results_dict
    


 

class CaPa_MoE_without_clinical(CLAM_SB):
    """
    Modality-level Dynamic MoE MIL
      - Expert 1: 用 virchow (x1) 特征
      - Expert 2: 用 [virchow, UNI] 拼接特征
      - Expert 3: 用 UNI (x2) 特征
      - Gating G(x): 对 {E1, E2, E3} 产生软权重
    三个 expert 都是平进平出的 MLP，最后动态融合三个特征，再接一个分类头。
    """
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super().__init__()
        # embed_dim 是投影后的特征维度
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        sz = self.size_dict[size_arg]

        # --------- 懒初始化的模态投影层（例如 virchow 2560 -> 1024）---------
        self.embed_dim = embed_dim
        self.proj_virchow = nn.Linear(2560, self.embed_dim)  # 需要时也可以给 UNI 做投影

        # --------- 每个模态自己的 attention pooling ---------
        Attn = Attn_Net_Gated if gate else Attn_Net
        self.attn_virchow = nn.Sequential(
            nn.Linear(sz[0], sz[1]), nn.ReLU(), nn.Dropout(dropout),
            Attn(L=sz[1], D=sz[2], dropout=dropout, n_classes=n_classes)
        )
        self.attn_uni = nn.Sequential(
            nn.Linear(sz[0], sz[1]), nn.ReLU(), nn.Dropout(dropout),
            Attn(L=sz[1], D=sz[2], dropout=dropout, n_classes=n_classes)
        )

        H = sz[1]   # 例如 512

        # --------- 三个 Expert：MLP 平进平出 ---------
        # E1: virchow 特征 [K, H] -> [K, H]
        self.expert1 = ExpertMLP(in_dim=H, hidden_dim=2 * H)       # 512 -> 1024 -> 512
        # E3: UNI 特征 [K, H] -> [K, H]
        self.expert3 = ExpertMLP(in_dim=H, hidden_dim=2 * H)
        # E2: 拼接特征 [K, 2H] -> [K, 2H] -> 最后映射回 H
        # 这里按照“拼接的专家大一点”设置：1024 -> 2048 -> 1024 -> ReLU
        self.expert2 = ExpertMLP(in_dim=2 * H, hidden_dim=4 * H)

        # 为了三路融合后维度一致，再加一层把 concat expert 输出压回 H
        self.expert2_out_proj = nn.Linear(2 * H, H)

        # --------- gating 网络：产生 3 个 expert 的权重 ---------
        # gate 输入改为包含 clinical_emb: g_path(2H) + clinical_emb(H) = 3H
        self.gate = nn.Sequential(
            nn.Linear(2 * H, sz[2]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(sz[2], 3)  # 3 个专家
        )

        # --------- bag-level 分类头（对融合后的特征做分类） ---------
        self.classifiers = nn.ModuleList([nn.Linear(H, 1) for _ in range(n_classes)])

        # --------- 实例级分类器，保留原逻辑：对两种模态分别做 top-k 监督 ---------
        self.instance_classifiers_virchow = nn.ModuleList([nn.Linear(H, 2) for _ in range(n_classes)])
        self.instance_classifiers_uni = nn.ModuleList([nn.Linear(H, 2) for _ in range(n_classes)])

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def _clam_pool(self, attn_net, h_in):
        """
        CLAM-style attention pooling
        attn_net: [Linear -> ReLU -> Dropout -> Attn_Net(_Gated)]
                  其中 Attn_Net(h_proj) 返回 (A, h_proj)
        h_in: [N, D]
        返回:
          M      : [K, H]
          A_raw  : [K, N]
          h_proj : [N, H]
        """
        A, h_proj = attn_net(h_in)      # A: [N, K], h_proj: [N, H]
        A = A.transpose(1, 0)           # [K, N]
        A_raw = A
        A = F.softmax(A, dim=1)         # softmax over N
        M = torch.mm(A, h_proj)         # [K, H]
        return M, A_raw, h_proj

    @torch.no_grad()
    def _global_summary(self, M_virchow, M_uni):
        """
        gating 使用的全局表征：对 per-class pooled features 做均值，
        然后拼接两个模态 -> [2H]
        """
        g1 = M_virchow.mean(dim=0)  # [H]
        g2 = M_uni.mean(dim=0)      # [H]
        return torch.cat([g1, g2], dim=-1)  # [2H]




    def forward(self, h_virchow, h_UNI, clinical, label=None, instance_eval=False,
                return_features=False, attention_only=False):
        """
        h_virchow: [N1, Din_v]
        h_UNI    : [N2, Din_u]
        """
        c = clinical
        device = h_virchow.device

        # ---- 懒初始化投影层（只建一次）----
        h_virchow = self.proj_virchow(h_virchow)  # [N1, embed_dim]

        # ---- 两个模态分别做 attention pooling ----
        M_virchow, A_virchow_raw, H_virchow = self._clam_pool(self.attn_virchow, h_virchow)
        M_uni, A_uni_raw, H_uni = self._clam_pool(self.attn_uni, h_UNI)

        if attention_only:
            return {"A_virchow": A_virchow_raw, "A_uni": A_uni_raw}

        H = M_virchow.size(1)  # 例如 512

        # ---- 三个 expert 产生特征 ----
        feat1 = self.expert1(M_virchow)                     # [K, H]  virchow
        feat3 = self.expert3(M_uni)                         # [K, H]  UNI
        feat2_in = torch.cat([M_virchow, M_uni], dim=-1)    # [K, 2H]
        feat2_big = self.expert2(feat2_in)                  # [K, 2H]
        feat2 = self.expert2_out_proj(feat2_big)            # [K, H]

        # ---- gating：soft 概率 + hard one-hot + 专家平衡正则 ----
        g_path = self._global_summary(M_virchow, M_uni)        # [2H]
        # 如果这个版本没有 clinical_emb，就直接用 g_path；有的话就像你写的拼上去：
        # g_in = torch.cat([g_path, clinical_emb], dim=-1)      # [2H + Hc]
        g_in = g_path

        g_logits = self.gate(g_in.unsqueeze(0))                # [1, 3]

        # soft 概率（可微），用于 gate_loss
        p_soft = F.softmax(g_logits, dim=-1).squeeze(0)        # [3]

        # 专家平衡：KL(p || uniform)，鼓励 p_soft 接近 [1/3,1/3,1/3]
        n_expert = p_soft.numel()                              # =3
        gate_loss = (p_soft * torch.log(p_soft * n_expert + 1e-8)).sum()

        # hard one-hot，用于真正路由 + instance_eval
        active_idx = torch.argmax(p_soft, dim=-1)              # 标量 0/1/2
        g = F.one_hot(active_idx, num_classes=n_expert).float()  # [3]



        

        # ---- 动态融合三个 expert 的特征（本质就是把被选中的那个拿出来）----
        # 因为 g 是 one-hot，这里就等价于选中某一个 expert 的输出
        F_mix = g[0] * feat1 + g[1] * feat2 + g[2] * feat3  # [K, H]

        # ---- 分类头：对融合后的特征做分类 ----
        logits = torch.empty(1, self.n_classes, device=F_mix.device, dtype=torch.float32)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](F_mix[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}

        # ---- 实例级监督：谁激活就谁做；拼接专家激活则两路都做 ----
        if instance_eval:
            total_inst_loss = 0.0
            all_preds, all_targets = [], []

            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # [K]
            active_expert = int(active_idx.item())  # 0: E1(virchow), 1: E2(concat), 2: E3(UNI)

            for i in range(self.n_classes):
                inst_label = int(inst_labels[i].item())

                # 先设默认值，避免没有分支参与时报未定义
                loss_v = loss_u = 0.0
                preds_v = preds_u = []
                t_v = t_u = []

                # ------ virchow 分支：expert1 或 expert2 激活才做 ------
                if active_expert in (0, 1):
                    clf_v = self.instance_classifiers_virchow[i]
                    if inst_label == 1:
                        loss_v, preds_v, t_v = self.inst_eval(A_virchow_raw[i], H_virchow, clf_v)
                    else:
                        if self.subtyping:
                            loss_v, preds_v, t_v = self.inst_eval_out(A_virchow_raw[i], H_virchow, clf_v)

                # ------ UNI 分支：expert3 或 expert2 激活才做 ------
                if active_expert in (1, 2):
                    clf_u = self.instance_classifiers_uni[i]
                    if inst_label == 1:
                        loss_u, preds_u, t_u = self.inst_eval(A_uni_raw[i], H_uni, clf_u)
                    else:
                        if self.subtyping:
                            loss_u, preds_u, t_u = self.inst_eval_out(A_uni_raw[i], H_uni, clf_u)

                # 累加 loss
                if isinstance(loss_v, torch.Tensor):
                    total_inst_loss = total_inst_loss + loss_v
                if isinstance(loss_u, torch.Tensor):
                    total_inst_loss = total_inst_loss + loss_u

                # 收集预测
                if len(preds_v):
                    all_preds.extend(preds_v.detach().cpu().numpy().tolist())
                    all_targets.extend(t_v.detach().cpu().numpy().tolist())
                if len(preds_u):
                    all_preds.extend(preds_u.detach().cpu().numpy().tolist())
                    all_targets.extend(t_u.detach().cpu().numpy().tolist())

            if self.subtyping:
                total_inst_loss = total_inst_loss / self.n_classes

            results_dict.update({
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds),

                # hard one-hot gate（用于统计）
                'gate': g.detach().cpu().numpy(),
                'active_expert': active_expert,

                # 额外返回 soft 概率（可选，用来画分布曲线）
                'gate_prob': p_soft.detach().cpu().numpy(),

                # ⭐ 专家平衡损失（KL 到均匀）
                'gate_loss': gate_loss,
            })

        if return_features:
            results_dict.update({
                'M_virchow': M_virchow,
                'M_uni': M_uni,
                'feat1': feat1,
                'feat2': feat2,
                'feat3': feat3,
                'F_mix': F_mix,
            })

        return logits, Y_prob, Y_hat, {'A_virchow': A_virchow_raw, 'A_uni': A_uni_raw}, results_dict




class CaPa_MoE_clinical_MLP(CLAM_SB):
    """
    Modality-level Dynamic MoE MIL
      - Expert 1: 用 virchow (x1) 特征
      - Expert 2: 用 [virchow, UNI] 拼接特征
      - Expert 3: 用 UNI (x2) 特征
      - Gating G(x): 对 {E1, E2, E3} 产生软权重
    三个 expert 都是平进平出的 MLP，最后动态融合三个特征，再接一个分类头。
    """
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024, clinical_dim=6):
        super().__init__()
        # embed_dim 是投影后的特征维度
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        sz = self.size_dict[size_arg]
        

        # --------- 懒初始化的模态投影层（例如 virchow 2560 -> 1024）---------
        self.embed_dim = embed_dim
        self.proj_virchow = nn.Linear(2560, self.embed_dim)  # 需要时也可以给 UNI 做投影

        # --------- 每个模态自己的 attention pooling ---------
        Attn = Attn_Net_Gated if gate else Attn_Net
        self.attn_virchow = nn.Sequential(
            nn.Linear(sz[0], sz[1]), nn.ReLU(), nn.Dropout(dropout),
            Attn(L=sz[1], D=sz[2], dropout=dropout, n_classes=n_classes)
        )
        self.attn_uni = nn.Sequential(
            nn.Linear(sz[0], sz[1]), nn.ReLU(), nn.Dropout(dropout),
            Attn(L=sz[1], D=sz[2], dropout=dropout, n_classes=n_classes)
        )

        H = sz[1]   # 例如 512


        # --------- 三个 Expert：MLP 平进平出 ---------
        # E1: virchow 特征 [K, H] -> [K, H]
        self.expert1 = ExpertMLP(in_dim=H, hidden_dim=2 * H)       # 512 -> 1024 -> 512
        # E3: UNI 特征 [K, H] -> [K, H]
        self.expert3 = ExpertMLP(in_dim=H, hidden_dim=2 * H)
        # E2: 拼接特征 [K, 2H] -> [K, 2H] -> 最后映射回 H
        # 这里按照“拼接的专家大一点”设置：1024 -> 2048 -> 1024 -> ReLU
        self.expert2 = ExpertMLP(in_dim=2 * H, hidden_dim=4 * H)

        # 为了三路融合后维度一致，再加一层把 concat expert 输出压回 H
        self.expert2_out_proj = nn.Linear(2 * H, H)
        self.fusion_proj = nn.Linear(2 * H, H)

        # --------- gating 网络：产生 3 个 expert 的权重 ---------
        self.gate = nn.Sequential(
            nn.Linear(3 * H, sz[2]),  # ⭐ 改成 3 * H
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sz[2], 3)       # 3 个专家
        )

        # --------- bag-level 分类头（对融合后的特征做分类） ---------
       

        # --------- 实例级分类器，保留原逻辑：对两种模态分别做 top-k 监督 ---------
        self.instance_classifiers_virchow = nn.ModuleList([nn.Linear(H, 2) for _ in range(n_classes)])
        self.instance_classifiers_uni = nn.ModuleList([nn.Linear(H, 2) for _ in range(n_classes)])

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        # --------- bag-level 分类头（对融合后的特征做分类） ---------
        
        self.clinical_dim = clinical_dim
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, H),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifiers = nn.ModuleList([nn.Linear(H, 1) for _ in range(n_classes)])
        self.tau = 1.0

    def _clam_pool(self, attn_net, h_in):
        """
        CLAM-style attention pooling
        attn_net: [Linear -> ReLU -> Dropout -> Attn_Net(_Gated)]
                  其中 Attn_Net(h_proj) 返回 (A, h_proj)
        h_in: [N, D]
        返回:
          M      : [K, H]
          A_raw  : [K, N]
          h_proj : [N, H]
        """
        A, h_proj = attn_net(h_in)      # A: [N, K], h_proj: [N, H]
        A = A.transpose(1, 0)           # [K, N]
        A_raw = A
        A = F.softmax(A, dim=1)         # softmax over N
        M = torch.mm(A, h_proj)         # [K, H]
        return M, A_raw, h_proj

    @torch.no_grad()
    def _global_summary(self, M_virchow, M_uni):
        """
        gating 使用的全局表征：对 per-class pooled features 做均值，
        然后拼接两个模态 -> [2H]
        """
        g1 = M_virchow.mean(dim=0)  # [H]
        g2 = M_uni.mean(dim=0)      # [H]
        return torch.cat([g1, g2], dim=-1)  # [2H]
    
    def _gumbel_softmax(self, logits, tau=1.0, hard=True, eps=1e-20):
        """
        logits: [B, 3]
        返回:
        y: [B, 3]  (如果 hard=True，forward 是 one-hot，backward 走 soft)
        idx: [B]   argmax 索引
        """
        # 1) 采样 Gumbel 噪声
        U = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(U + eps) + eps)

        # 2) 加噪声后做 softmax
        y = F.softmax((logits + gumbel) / tau, dim=-1)

        if hard:
            # 3) straight-through: forward one-hot, backward soft
            idx = y.max(dim=-1, keepdim=True)[1]      # [B,1]
            y_hard = torch.zeros_like(y).scatter_(1, idx, 1.0)
            # y_hard - y.detach() + y: forward 用 y_hard，梯度来自 y
            y = y_hard.detach() - y.detach() + y
            return y, idx.squeeze(-1)
        else:
            return y, None



    def forward(self, h_virchow, h_UNI, clinical, label=None, instance_eval=False,
                return_features=False, attention_only=False):
        """
        h_virchow: [N1, Din_v]
        h_UNI    : [N2, Din_u]
        """
        device = h_virchow.device

        # ---- 懒初始化投影层（只建一次）----
        h_virchow = self.proj_virchow(h_virchow)  # [N1, embed_dim]

        # ---- 两个模态分别做 attention pooling ----
        M_virchow, A_virchow_raw, H_virchow = self._clam_pool(self.attn_virchow, h_virchow)
        M_uni, A_uni_raw, H_uni = self._clam_pool(self.attn_uni, h_UNI)

        if attention_only:
            return {"A_virchow": A_virchow_raw, "A_uni": A_uni_raw}

        H = M_virchow.size(1)  # 例如 512

        # ---- clinical embedding（统一算一次，后面 gating 和融合都用它）----
        if clinical is None:
            # 没有临床，用 0 向量占位
            clinical_emb = torch.zeros(H, device=device)
        else:
            if clinical.dim() == 1:
                c_vec = clinical.unsqueeze(0)      # [1, D]
            else:
                c_vec = clinical                   # [1, D] 或 [B, D]
            c_vec = c_vec.to(device).float()
            clinical_emb = self.clinical_mlp(c_vec).squeeze(0)   # [H]

        # ---- 三个 expert 产生特征 ----
        feat1 = self.expert1(M_virchow)                     # [K, H]  virchow
        feat3 = self.expert3(M_uni)                         # [K, H]  UNI
        feat2_in = torch.cat([M_virchow, M_uni], dim=-1)    # [K, 2H]
        feat2_big = self.expert2(feat2_in)                  # [K, 2H]
        feat2 = self.expert2_out_proj(feat2_big)            # [K, H]

        # ---- gating 使用的全局表征 + 临床 ----
        g_path = self._global_summary(M_virchow, M_uni)     # [2H]
        g_in = torch.cat([g_path, clinical_emb], dim=-1)    # [3H]
        g_logits = self.gate(g_in.unsqueeze(0))             # [1, 3]

        if self.training:
            # 训练阶段：hard=True，forward one-hot，backward soft
            g_st, active_idx = self._gumbel_softmax(
                g_logits,
                tau=self.tau,
                hard=True
            )                                                # g_st: [1,3]
            g = g_st.squeeze(0)                              # [3]
        else:
            # 推理阶段：直接 argmax one-hot
            active_idx = torch.argmax(g_logits, dim=-1)      # [1]
            g = F.one_hot(active_idx, num_classes=3).float().squeeze(0)  # [3]

        # ---- 动态融合三个 expert 的特征 ----
        # g 是 one-hot，本质上选中了某一个 expert 的输出
        F_mix = g[0] * feat1 + g[1] * feat2 + g[2] * feat3   # [K, H]

        # ---- 融合 F_mix 和 clinical_emb ----
        F_fused = []
        for cls in range(self.n_classes):
            # 这里按类重复使用同一个 clinical_emb，如果你以后想 class-specific clinical 也可以扩展
            joint_c = torch.cat([F_mix[cls], clinical_emb], dim=-1)   # [2H]
            fused_c = self.fusion_proj(joint_c)                       # [H]
            F_fused.append(fused_c)
        F_fused = torch.stack(F_fused, dim=0)                         # [C, H]

        # ---- 分类头：对融合后的特征做分类 ----
        logits = torch.empty(1, self.n_classes, device=F_fused.device, dtype=torch.float32)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](F_fused[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {}

        # ---- 实例级监督：谁激活就谁做；拼接专家激活则两路都做 ----
        if instance_eval:
            total_inst_loss = 0.0
            all_preds, all_targets = [], []

            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # [K]
            active_expert = int(active_idx.item())  # 0: E1(virchow), 1: E2(concat), 2: E3(UNI)

            for i in range(self.n_classes):
                inst_label = int(inst_labels[i].item())

                # 先设默认值，避免没有分支参与时报未定义
                loss_v = loss_u = 0.0
                preds_v = preds_u = []
                t_v = t_u = []

                # ------ virchow 分支：expert1 或 expert2 激活才做 ------
                if active_expert in (0, 1):
                    clf_v = self.instance_classifiers_virchow[i]
                    if inst_label == 1:
                        loss_v, preds_v, t_v = self.inst_eval(A_virchow_raw[i], H_virchow, clf_v)
                    else:
                        if self.subtyping:
                            loss_v, preds_v, t_v = self.inst_eval_out(A_virchow_raw[i], H_virchow, clf_v)

                # ------ UNI 分支：expert3 或 expert2 激活才做 ------
                if active_expert in (1, 2):
                    clf_u = self.instance_classifiers_uni[i]
                    if inst_label == 1:
                        loss_u, preds_u, t_u = self.inst_eval(A_uni_raw[i], H_uni, clf_u)
                    else:
                        if self.subtyping:
                            loss_u, preds_u, t_u = self.inst_eval_out(A_uni_raw[i], H_uni, clf_u)

                # 累加 loss
                if isinstance(loss_v, torch.Tensor):
                    total_inst_loss = total_inst_loss + loss_v
                if isinstance(loss_u, torch.Tensor):
                    total_inst_loss = total_inst_loss + loss_u

                # 收集预测
                if len(preds_v):
                    all_preds.extend(preds_v.detach().cpu().numpy().tolist())
                    all_targets.extend(t_v.detach().cpu().numpy().tolist())
                if len(preds_u):
                    all_preds.extend(preds_u.detach().cpu().numpy().tolist())
                    all_targets.extend(t_u.detach().cpu().numpy().tolist())

            if self.subtyping:
                total_inst_loss = total_inst_loss / self.n_classes

            results_dict.update({
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds),
                'gate': g.detach().cpu().numpy(),      # 现在是 one-hot
                'active_expert': active_expert,        # 0/1/2
            })

        if return_features:
            results_dict.update({
                'M_virchow': M_virchow,
                'M_uni': M_uni,
                'feat1': feat1,
                'feat2': feat2,
                'feat3': feat3,
                'F_fused': F_fused,
            })

        return logits, Y_prob, Y_hat, {'A_virchow': A_virchow_raw, 'A_uni': A_uni_raw}, results_dict
    



class CaPa_MoE_clinical_gate(CLAM_SB):
    """
    Modality-level Dynamic MoE MIL
      - Expert 1: 用 virchow (x1) 特征
      - Expert 2: 用 [virchow, UNI] 拼接特征
      - Expert 3: 用 UNI (x2) 特征
      - Gating G(x): 对 {E1, E2, E3} 产生软权重
    三个 expert 都是平进平出的 MLP，最后动态融合三个特征，再接一个分类头。
    """
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024, clinical_dim=6):
        super().__init__()
        # embed_dim 是投影后的特征维度
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        sz = self.size_dict[size_arg]
        

        # --------- 懒初始化的模态投影层（例如 virchow 2560 -> 1024）---------
        self.embed_dim = embed_dim
        self.proj_virchow = nn.Linear(2560, self.embed_dim)  # 需要时也可以给 UNI 做投影

        # --------- 每个模态自己的 attention pooling ---------
        Attn = Attn_Net_Gated if gate else Attn_Net
        self.attn_virchow = nn.Sequential(
            nn.Linear(sz[0], sz[1]), nn.ReLU(), nn.Dropout(dropout),
            Attn(L=sz[1], D=sz[2], dropout=dropout, n_classes=n_classes)
        )
        self.attn_uni = nn.Sequential(
            nn.Linear(sz[0], sz[1]), nn.ReLU(), nn.Dropout(dropout),
            Attn(L=sz[1], D=sz[2], dropout=dropout, n_classes=n_classes)
        )

        H = sz[1]   # 例如 512
        self.gate = nn.Linear(2 * H, 3) 

        # --------- 三个 Expert：MLP 平进平出 ---------
        # E1: virchow 特征 [K, H] -> [K, H]
        self.expert1 = ExpertMLP(in_dim=H, hidden_dim=2 * H)       # 512 -> 1024 -> 512
        # E3: UNI 特征 [K, H] -> [K, H]
        self.expert3 = ExpertMLP(in_dim=H, hidden_dim=2 * H)
        # E2: 拼接特征 [K, 2H] -> [K, 2H] -> 最后映射回 H
        # 这里按照“拼接的专家大一点”设置：1024 -> 2048 -> 1024 -> ReLU
        self.expert2 = ExpertMLP(in_dim=2 * H, hidden_dim=4 * H)

        # 为了三路融合后维度一致，再加一层把 concat expert 输出压回 H
        self.expert2_out_proj = nn.Linear(2 * H, H)
        self.fusion_proj = nn.Linear(2 * H, H)

        # --------- gating 网络：产生 3 个 expert 的权重 ---------
        self.gate = nn.Sequential(
            nn.Linear(3 * H, sz[2]),   # 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sz[2], 3)
        )

        # --------- bag-level 分类头（对融合后的特征做分类） ---------
       

        # --------- 实例级分类器，保留原逻辑：对两种模态分别做 top-k 监督 ---------
        self.instance_classifiers_virchow = nn.ModuleList([nn.Linear(H, 2) for _ in range(n_classes)])
        self.instance_classifiers_uni = nn.ModuleList([nn.Linear(H, 2) for _ in range(n_classes)])

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        # --------- bag-level 分类头（对融合后的特征做分类） ---------
        
        self.clinical_dim = clinical_dim
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, H),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifiers = nn.ModuleList([nn.Linear(H, 1) for _ in range(n_classes)])
        self.tau = 1.0

    def _clam_pool(self, attn_net, h_in):
        """
        CLAM-style attention pooling
        attn_net: [Linear -> ReLU -> Dropout -> Attn_Net(_Gated)]
                  其中 Attn_Net(h_proj) 返回 (A, h_proj)
        h_in: [N, D]
        返回:
          M      : [K, H]
          A_raw  : [K, N]
          h_proj : [N, H]
        """
        A, h_proj = attn_net(h_in)      # A: [N, K], h_proj: [N, H]
        A = A.transpose(1, 0)           # [K, N]
        A_raw = A
        A = F.softmax(A, dim=1)         # softmax over N
        M = torch.mm(A, h_proj)         # [K, H]
        return M, A_raw, h_proj

    @torch.no_grad()
    def _global_summary(self, M_virchow, M_uni):
        """
        gating 使用的全局表征：对 per-class pooled features 做均值，
        然后拼接两个模态 -> [2H]
        """
        g1 = M_virchow.mean(dim=0)  # [H]
        g2 = M_uni.mean(dim=0)      # [H]
        return torch.cat([g1, g2], dim=-1)  # [2H]
    
    def _gumbel_softmax(self, logits, tau=1.0, hard=True, eps=1e-20):
        """
        logits: [B, 3]
        返回:
        y: [B, 3]  (如果 hard=True，forward 是 one-hot，backward 走 soft)
        idx: [B]   argmax 索引
        """
        # 1) 采样 Gumbel 噪声
        U = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(U + eps) + eps)

        # 2) 加噪声后做 softmax
        y = F.softmax((logits + gumbel) / tau, dim=-1)

        if hard:
            # 3) straight-through: forward one-hot, backward soft
            idx = y.max(dim=-1, keepdim=True)[1]      # [B,1]
            y_hard = torch.zeros_like(y).scatter_(1, idx, 1.0)
            # y_hard - y.detach() + y: forward 用 y_hard，梯度来自 y
            y = y_hard.detach() - y.detach() + y
            return y, idx.squeeze(-1)
        else:
            return y, None



    def forward(self, h_virchow, h_UNI, clinical, label=None, instance_eval=False,
                return_features=False, attention_only=False):
        """
        h_virchow: [N1, Din_v]
        h_UNI    : [N2, Din_u]
        """
        
        device = h_virchow.device

        # ---- 懒初始化投影层（只建一次）----
        h_virchow = self.proj_virchow(h_virchow)  # [N1, embed_dim]

        # ---- 两个模态分别做 attention pooling ----
        M_virchow, A_virchow_raw, H_virchow = self._clam_pool(self.attn_virchow, h_virchow)
        M_uni, A_uni_raw, H_uni = self._clam_pool(self.attn_uni, h_UNI)

        if attention_only:
            return {"A_virchow": A_virchow_raw, "A_uni": A_uni_raw}

        H = M_virchow.size(1)  # 例如 512

        if clinical.dim() == 1:
            c_vec = clinical.unsqueeze(0)   # [1, D]
        else:
            c_vec = clinical  

        c_vec = c_vec.to(device).float()
        clinical_emb = self.clinical_mlp(c_vec)   # [1, H]
        clinical_emb = clinical_emb.squeeze(0)

        # ---- 三个 expert 产生特征 ----
        feat1 = self.expert1(M_virchow)                     # [K, H]  virchow
        feat3 = self.expert3(M_uni)                         # [K, H]  UNI
        feat2_in = torch.cat([M_virchow, M_uni], dim=-1)    # [K, 2H]
        feat2_big = self.expert2(feat2_in)                  # [K, 2H]
        feat2 = self.expert2_out_proj(feat2_big)            # [K, H]


        g_path = self._global_summary(M_virchow, M_uni)        # [2H]

        g_in = torch.cat([g_path, clinical_emb], dim=-1)
        g_logits = self.gate(g_in.unsqueeze(0))              # [1, 3]

        if self.training:
            # 训练阶段：hard=True，forward one-hot，backward soft
            g_st, active_idx = self._gumbel_softmax(
                g_logits,
                tau=self.tau,
                hard=True
            )                                                # g_st: [1,3]
            g = g_st.squeeze(0)                              # [3]
        else:
            # 推理阶段：直接 argmax one-hot
            active_idx = torch.argmax(g_logits, dim=-1)      # [1]
            g = F.one_hot(active_idx, num_classes=3).float().squeeze(0)  # [3]

        
        F_mix = g[0] * feat1 + g[1] * feat2 + g[2] * feat3  # [K, H]



        

        
        F_fused = []
        for c in range(self.n_classes):
            joint_c = torch.cat([F_mix[c], clinical_emb], dim=-1)   # [2H]
            fused_c = self.fusion_proj(joint_c)                     # [H]
            F_fused.append(fused_c)
        F_fused = torch.stack(F_fused, dim=0)


        # ---- 分类头：对融合后的特征做分类 ----
        logits = torch.empty(1, self.n_classes, device=F_fused.device, dtype=torch.float32)
        # logits = torch.empty(1, self.n_classes, device=F_mix.device, dtype=torch.float32)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](F_fused[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}

        # ---- 实例级监督：谁激活就谁做；拼接专家激活则两路都做 ----
        if instance_eval:
            total_inst_loss = 0.0
            all_preds, all_targets = [], []

            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # [K]
            active_expert = int(active_idx.item())  # 0: E1(virchow), 1: E2(concat), 2: E3(UNI)

            for i in range(self.n_classes):
                inst_label = int(inst_labels[i].item())

                # 先设默认值，避免没有分支参与时报未定义
                loss_v = loss_u = 0.0
                preds_v = preds_u = []
                t_v = t_u = []

                # ------ virchow 分支：expert1 或 expert2 激活才做 ------
                if active_expert in (0, 1):
                    clf_v = self.instance_classifiers_virchow[i]
                    if inst_label == 1:
                        loss_v, preds_v, t_v = self.inst_eval(A_virchow_raw[i], H_virchow, clf_v)
                    else:
                        if self.subtyping:
                            loss_v, preds_v, t_v = self.inst_eval_out(A_virchow_raw[i], H_virchow, clf_v)

                # ------ UNI 分支：expert3 或 expert2 激活才做 ------
                if active_expert in (1, 2):
                    clf_u = self.instance_classifiers_uni[i]
                    if inst_label == 1:
                        loss_u, preds_u, t_u = self.inst_eval(A_uni_raw[i], H_uni, clf_u)
                    else:
                        if self.subtyping:
                            loss_u, preds_u, t_u = self.inst_eval_out(A_uni_raw[i], H_uni, clf_u)

                # 累加 loss
                if isinstance(loss_v, torch.Tensor):
                    total_inst_loss = total_inst_loss + loss_v
                if isinstance(loss_u, torch.Tensor):
                    total_inst_loss = total_inst_loss + loss_u

                # 收集预测
                if len(preds_v):
                    all_preds.extend(preds_v.detach().cpu().numpy().tolist())
                    all_targets.extend(t_v.detach().cpu().numpy().tolist())
                if len(preds_u):
                    all_preds.extend(preds_u.detach().cpu().numpy().tolist())
                    all_targets.extend(t_u.detach().cpu().numpy().tolist())

            if self.subtyping:
                total_inst_loss = total_inst_loss / self.n_classes

            results_dict.update({
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds),
                'gate': g.detach().cpu().numpy(),      # 现在是 one-hot
                'active_expert': active_expert,        # 0/1/2
            })

        if return_features:
            results_dict.update({
                'M_virchow': M_virchow,
                'M_uni': M_uni,
                'feat1': feat1,
                'feat2': feat2,
                'feat3': feat3,
                'F_fused': F_fused,
            })

        return logits, Y_prob, Y_hat, {'A_virchow': A_virchow_raw, 'A_uni': A_uni_raw}, results_dict
    

class CaPa_MoE(CLAM_SB):
    """
    Modality-level Dynamic MoE MIL
      - Expert 1: 用 virchow (x1) 特征
      - Expert 2: 用 [virchow, UNI] 拼接特征
      - Expert 3: 用 UNI (x2) 特征
      - Gating G(x): 对 {E1, E2, E3} 产生软权重
    三个 expert 都是平进平出的 MLP，最后动态融合三个特征，再接一个分类头。
    """
    def __init__(self, gate=True, size_arg="small", dropout=0., k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024, clinical_dim=6):
        super().__init__()
        # embed_dim 是投影后的特征维度
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        sz = self.size_dict[size_arg]
        

        # --------- 懒初始化的模态投影层（例如 virchow 2560 -> 1024）---------
        self.embed_dim = embed_dim
        self.proj_virchow = nn.Linear(2560, self.embed_dim)  # 需要时也可以给 UNI 做投影

        # --------- 每个模态自己的 attention pooling ---------
        Attn = Attn_Net_Gated if gate else Attn_Net
        self.attn_virchow = nn.Sequential(
            nn.Linear(sz[0], sz[1]), nn.ReLU(), nn.Dropout(dropout),
            Attn(L=sz[1], D=sz[2], dropout=dropout, n_classes=n_classes)
        )
        self.attn_uni = nn.Sequential(
            nn.Linear(sz[0], sz[1]), nn.ReLU(), nn.Dropout(dropout),
            Attn(L=sz[1], D=sz[2], dropout=dropout, n_classes=n_classes)
        )

        H = sz[1]   # 例如 512
        self.gate = nn.Linear(2 * H, 3) 

        # --------- 三个 Expert：MLP 平进平出 ---------
        # E1: virchow 特征 [K, H] -> [K, H]
        self.expert1 = ExpertMLP(in_dim=H, hidden_dim=2 * H)       # 512 -> 1024 -> 512
        # E3: UNI 特征 [K, H] -> [K, H]
        self.expert3 = ExpertMLP(in_dim=H, hidden_dim=2 * H)
        # E2: 拼接特征 [K, 2H] -> [K, 2H] -> 最后映射回 H
        # 这里按照“拼接的专家大一点”设置：1024 -> 2048 -> 1024 -> ReLU
        self.expert2 = ExpertMLP(in_dim=2 * H, hidden_dim=4 * H)

        # 为了三路融合后维度一致，再加一层把 concat expert 输出压回 H
        self.expert2_out_proj = nn.Linear(2 * H, H)
        self.fusion_proj = nn.Linear(2 * H, H)

        # --------- gating 网络：产生 3 个 expert 的权重 ---------
        self.gate = nn.Sequential(
            nn.Linear(3 * H, sz[2]),   # 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sz[2], 3)
        )

        # --------- bag-level 分类头（对融合后的特征做分类） ---------
       

        # --------- 实例级分类器，保留原逻辑：对两种模态分别做 top-k 监督 ---------
        self.instance_classifiers_virchow = nn.ModuleList([nn.Linear(H, 2) for _ in range(n_classes)])
        self.instance_classifiers_uni = nn.ModuleList([nn.Linear(H, 2) for _ in range(n_classes)])

        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        # --------- bag-level 分类头（对融合后的特征做分类） ---------
        
        self.clinical_dim = clinical_dim
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, H),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifiers = nn.ModuleList([nn.Linear(H, 1) for _ in range(n_classes)])



        self.tau = 1.0
        self.soft_epochs = 0       # 纯 softmax 阶段
        self.hard_start = 0       # 从这个 epoch 开始 hard one-hot
        self.tau_start = 1.0
        self.tau_end = 0.1
        self.tau_anneal_epochs = 40
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """在训练循环中每个 epoch 开头调用，用来更新 current_epoch 和 tau。"""
        self.current_epoch = epoch

        # 线性退火 tau（你也可以改成指数、cosine 等）
        t = max(0.0, min(1.0, epoch / float(self.tau_anneal_epochs)))
        self.tau = self.tau_start * (1 - t) + self.tau_end * t

    def _clam_pool(self, attn_net, h_in):
        """
        CLAM-style attention pooling
        attn_net: [Linear -> ReLU -> Dropout -> Attn_Net(_Gated)]
                  其中 Attn_Net(h_proj) 返回 (A, h_proj)
        h_in: [N, D]
        返回:
          M      : [K, H]
          A_raw  : [K, N]
          h_proj : [N, H]
        """
        A, h_proj = attn_net(h_in)      # A: [N, K], h_proj: [N, H]
        A = A.transpose(1, 0)           # [K, N]
        A_raw = A
        A = F.softmax(A, dim=1)         # softmax over N
        M = torch.mm(A, h_proj)         # [K, H]
        return M, A_raw, h_proj

    @torch.no_grad()
    def _global_summary(self, M_virchow, M_uni):
        """
        gating 使用的全局表征：对 per-class pooled features 做均值，
        然后拼接两个模态 -> [2H]
        """
        g1 = M_virchow.mean(dim=0)  # [H]
        g2 = M_uni.mean(dim=0)      # [H]
        return torch.cat([g1, g2], dim=-1)  # [2H]
    
    def _gumbel_softmax(self, logits, tau=1.0, hard=True, eps=1e-20):
        """
        logits: [B, 3]
        返回:
            y:   [B, 3]
            idx: [B]
        """
        U = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(U + eps) + eps)
        y = F.softmax((logits + gumbel) / tau, dim=-1)  # [B,3]

        # 不管 hard 不 hard，都先算一份 idx
        idx = y.max(dim=-1)[1]  # [B]

        if hard:
            # straight-through
            y_hard = torch.zeros_like(y).scatter_(1, idx.unsqueeze(-1), 1.0)
            y = y_hard.detach() - y.detach() + y

        return y, idx



    def forward(self, h_virchow, h_UNI, clinical, label=None, instance_eval=False,
                return_features=False, attention_only=False):
        """
        h_virchow: [N1, Din_v]
        h_UNI    : [N2, Din_u]
        """
        
        device = h_virchow.device

        # ---- 懒初始化投影层（只建一次）----
        h_virchow = self.proj_virchow(h_virchow)  # [N1, embed_dim]

        # ---- 两个模态分别做 attention pooling ----
        M_virchow, A_virchow_raw, H_virchow = self._clam_pool(self.attn_virchow, h_virchow)
        M_uni, A_uni_raw, H_uni = self._clam_pool(self.attn_uni, h_UNI)

        if attention_only:
            return {"A_virchow": A_virchow_raw, "A_uni": A_uni_raw}

        H = M_virchow.size(1)  # 例如 512

        if clinical.dim() == 1:
            c_vec = clinical.unsqueeze(0)   # [1, D]
        else:
            c_vec = clinical  

        c_vec = c_vec.to(device).float()
        clinical_emb = self.clinical_mlp(c_vec)   # [1, H]
        clinical_emb = clinical_emb.squeeze(0)

        # ---- 三个 expert 产生特征 ----
        feat1 = self.expert1(M_virchow)                     # [K, H]  virchow
        feat3 = self.expert3(M_uni)                         # [K, H]  UNI
        feat2_in = torch.cat([M_virchow, M_uni], dim=-1)    # [K, 2H]
        feat2_big = self.expert2(feat2_in)                  # [K, 2H]
        feat2 = self.expert2_out_proj(feat2_big)            # [K, H]


        g_path = self._global_summary(M_virchow, M_uni)        # [2H]

        g_in = torch.cat([g_path, clinical_emb], dim=-1)
        g_logits = self.gate(g_in.unsqueeze(0))              # [1, 3]

        p_soft = F.softmax(g_logits, dim=-1).squeeze(0) 

        if self.current_epoch < self.soft_epochs:
            # 阶段 1：纯 softmax，连续加权
            g = p_soft      # [3]
            # 为了 instance_eval 里还能用 active_idx，取一个 argmax 但不影响前向
            active_idx = torch.argmax(g, dim=-1, keepdim=False)  # 标量
        else:
            # 阶段 2/3：Gumbel-Softmax + 温度退火
            # hard 在 early-anneal 期间可以先用 False，到了 hard_start 以后再 True
            use_hard = self.current_epoch >= self.hard_start

            g_st, active_idx = self._gumbel_softmax(
                g_logits,
                tau=self.tau,
                hard=use_hard
            )                                               # g_st: [1, 3]
            g = g_st.squeeze(0)                             # [3]
            # active_idx: [1] -> 标量
            active_idx = int(active_idx.item())

        
        F_mix = g[0] * feat1 + g[1] * feat2 + g[2] * feat3  # [K, H]



        

        
        F_fused = []
        for c in range(self.n_classes):
            joint_c = torch.cat([F_mix[c], clinical_emb], dim=-1)   # [2H]
            fused_c = self.fusion_proj(joint_c)                     # [H]
            F_fused.append(fused_c)
        F_fused = torch.stack(F_fused, dim=0)


        # ---- 分类头：对融合后的特征做分类 ----
        logits = torch.empty(1, self.n_classes, device=F_fused.device, dtype=torch.float32)
        # logits = torch.empty(1, self.n_classes, device=F_mix.device, dtype=torch.float32)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](F_fused[c])

        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}

        # ---- 实例级监督：谁激活就谁做；拼接专家激活则两路都做 ----
        if instance_eval:
            total_inst_loss = 0.0
            all_preds, all_targets = [], []

            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # [K]
            if isinstance(active_idx, torch.Tensor):
                active_expert = int(active_idx.detach().item())
            else:
                active_expert = int(active_idx)  # 已经是 int 了

            for i in range(self.n_classes):
                inst_label = int(inst_labels[i].item())

                # 先设默认值，避免没有分支参与时报未定义
                loss_v = loss_u = 0.0
                preds_v = preds_u = []
                t_v = t_u = []

                # ------ virchow 分支：expert1 或 expert2 激活才做 ------
                if active_expert in (0, 1):
                    clf_v = self.instance_classifiers_virchow[i]
                    if inst_label == 1:
                        loss_v, preds_v, t_v = self.inst_eval(A_virchow_raw[i], H_virchow, clf_v)
                    else:
                        if self.subtyping:
                            loss_v, preds_v, t_v = self.inst_eval_out(A_virchow_raw[i], H_virchow, clf_v)

                # ------ UNI 分支：expert3 或 expert2 激活才做 ------
                if active_expert in (1, 2):
                    clf_u = self.instance_classifiers_uni[i]
                    if inst_label == 1:
                        loss_u, preds_u, t_u = self.inst_eval(A_uni_raw[i], H_uni, clf_u)
                    else:
                        if self.subtyping:
                            loss_u, preds_u, t_u = self.inst_eval_out(A_uni_raw[i], H_uni, clf_u)

                # 累加 loss
                if isinstance(loss_v, torch.Tensor):
                    total_inst_loss = total_inst_loss + loss_v
                if isinstance(loss_u, torch.Tensor):
                    total_inst_loss = total_inst_loss + loss_u

                # 收集预测
                if len(preds_v):
                    all_preds.extend(preds_v.detach().cpu().numpy().tolist())
                    all_targets.extend(t_v.detach().cpu().numpy().tolist())
                if len(preds_u):
                    all_preds.extend(preds_u.detach().cpu().numpy().tolist())
                    all_targets.extend(t_u.detach().cpu().numpy().tolist())

            if self.subtyping:
                total_inst_loss = total_inst_loss / self.n_classes


            n_expert = p_soft.size(0)
            gate_loss = (p_soft * torch.log(p_soft * n_expert + 1e-8)).sum()

            results_dict.update({
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds),
                'gate': g.detach().cpu().numpy(),      # 用于日志
                'gate_prob': p_soft,                   # 保留带梯度的 soft gate
                'gate_loss': gate_loss,                # 带梯度的 gate 正则 
                'active_expert': active_expert,        # 0/1/2
            })




        if return_features:
            results_dict.update({
                'M_virchow': M_virchow,
                'M_uni': M_uni,
                'feat1': feat1,
                'feat2': feat2,
                'feat3': feat3,
                'F_fused': F_fused,
            })

        return logits, Y_prob, Y_hat, {'A_virchow': A_virchow_raw, 'A_uni': A_uni_raw}, results_dict