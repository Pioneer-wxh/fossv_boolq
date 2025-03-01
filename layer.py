import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class FOSSVLayer:
    def __init__(
        self,
        r: int,
        alpha: Optional[float],
        dropout: float,
        merge_weights: bool,  # 用来控制是否进行合并解合并
    ):
        super().__init__()
        self.r = r
        if alpha is None:
            self.scale = 1
        else:
            self.scale = alpha / r
        # Optional dropout
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False  # 这个布尔变量是用来表示内部状态，模型是否合并
        self.merge_weights = merge_weights
       

class Linear(nn.Module, FOSSVLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        merge_weights: bool = True,
        bias=False,
        fan_in_fan_out: bool = False,
    ):
        # 先初始化nn.Linear
        nn.Module.__init__(self)

        FOSSVLayer.__init__(
            self,
            r=r,
            alpha=alpha,
            dropout=dropout,
            merge_weights=merge_weights
        )

        # 创建必要的参数容器，便于后面的初始化储存原始linear层的参数
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))  # 初始化权重矩阵，初始化一个随机的权重矩阵并且转化为pytorch版本
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features)))  # 如果有bias的话，初始化一个bias，是一个一维张量
        else:
            self.bias = None

        self.fan_in_fan_out = fan_in_fan_out


        if r > 0:
            self.fossv_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.fossv_S = nn.Parameter(self.weight.new_zeros(r))
            self.fossv_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.weight.requires_grad = False
            self.scaling = self.scale

            # Additional attributes to store initial SVD decomposition
            self.u_small = None
            self.s_small = None
            self.vt_small = None

            # Dash part adapter preparation
            self.index = 5
            self.lora_index = nn.Parameter(self.weight.new_zeros(self.index))
            self.weight_u_top = nn.Parameter(self.weight.new_zeros(out_features, self.index))
            self.weight_vt_top = nn.Parameter(self.weight.new_zeros(self.index, in_features))

            self.warmup = 100
            self.FLAG = 0

    def fossv_init(
        self,
        weight_dtype: Optional[torch.dtype] = None,
        adapter_dtype: Optional[torch.dtype] = None,
    ):
        device = next(self.parameters()).device  # 获取当前模型的设备

        if weight_dtype is None:
            weight_dtype = self.weight.dtype
        if adapter_dtype is None:
            adapter_dtype = weight_dtype

        if hasattr(self, "fossv_A"):
            self.merged = False  # 当前模型未进行合并
            self.weight.to(torch.float32)

            # Perform SVD on the original weight matrix
            u, s, vt = torch.linalg.svd(self.weight.T, full_matrices=False)

            # Extract the smallest r singular values
            self.u_small = u[:, -self.r:].to(device)  # Store u_small on the correct device
            self.s_small = s[-self.r:].to(device)  # Store s_small on the correct device
            self.vt_small = vt[-self.r:, :].to(device)  # Store vt_small on the correct device

            # Fill adapter parameters
            self.fossv_A.data = self.u_small.T.contiguous().to(adapter_dtype)
            self.fossv_S.data = self.s_small.to(adapter_dtype)
            self.fossv_B.data = self.vt_small.T.contiguous().to(adapter_dtype)

            # Merge adapter parameters into the original weight matrix
            merge = (self.fossv_B @ torch.diag(self.fossv_S)) @ self.fossv_A
            self.weight.data = (self.weight - merge * self.scale).to(weight_dtype)

    def calculate_change_rate(self, a, bb, r):
        self.lora_change_a = nn.Parameter(a)
        self.lora_change_bb = nn.Parameter(bb)

        change_rate = abs(bb) / abs(a)
        _, top_r_indices = torch.topk(change_rate, r)
        return top_r_indices

    def _merge(self, mode: bool):  # mode表示是否要进行合并的动作，self.merged表示状态
        """
        Merge or unmerge SORSA weights with the main weight matrix.是否和主权重矩阵进行合并

        Args:
            mode (bool): If True, merge the weights. If False, unmerge the weights.根据mode变量决定如何进行操作
        """
        if mode:  # 合并
            if self.merge_weights and not self.merged:  # 如果要进行合并，且模型未进行合并
                # Merge the weights and mark it
                if self.r > 0:
                    merge = (self.fossv_B @ torch.diag(self.fossv_S)) @ self.fossv_A
                    self.weight.data += merge * self.scale
                self.merged = True
        else:  # 解合并
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    merge = (self.fossv_B @ torch.diag(self.fossv_S)) @ self.fossv_A
                    self.weight.data -= merge * self.scale
                self.merged = False

    def forward(self, x: torch.Tensor):
        device = next(self.parameters()).device  # 获取当前模型的设备

        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:  # r大于0，且当前模型未进行合并
            result = F.linear(x.to(device), T(self.weight).to(device), bias=self.bias)
            fossv_delta = (self.fossv_B @ torch.diag(self.fossv_S)) @ self.fossv_A * self.scaling
            fossv_delta = fossv_delta.to(device)  # 显式移动到当前设备
            result += self.dropout(x.to(device)) @ fossv_delta.T

            if self.FLAG < self.warmup:
                if self.FLAG == 0:
                    self.lora_index.requires_grad = False
                    self.weight_u_top.requires_grad = False
                    self.weight_vt_top.requires_grad = False
                self.FLAG += 1
                return result

            # layer.py 的 forward 方法
            elif self.FLAG == self.warmup:
                # 确保 u_small 和 vt_small 在正确的设备上
                u_small = self.u_small.to(device)
                vt_small = self.vt_small.to(device)
                
                # 计算 delta_sigma（确保 fossv_delta 在正确设备上）
                fossv_delta = fossv_delta.to(device)
                delta_sigma = torch.diag(torch.matmul(torch.matmul(u_small.T, fossv_delta), vt_small.T))
                
                # 确保 self.s_small 在正确设备上
                top_index = self.calculate_change_rate(self.s_small.to(device), delta_sigma, self.index)

                # Update SVD decomposition U and Vt matrices
                self.weight_u_top.data = u_small[:, top_index]
                self.weight_vt_top.data = vt_small[top_index, :]

                # Unfreeze dash part parameters
                self.lora_index.requires_grad = True
                self.FLAG += 1

            if self.FLAG > self.warmup:
                result += self.dropout(x.to(device)) @ (self.weight_u_top @ torch.diag(self.lora_index) @ self.weight_vt_top).T
                return result

        else:
            return F.linear(x.to(device), T(self.weight).to(device), bias=self.bias)




def calc_ortho(model):
    """
    Calculate the average orthogonal regularizer loss for SORSA matrices in the model.

    Args:
        model: The PyTorch model containing SORSA layers.

    Returns:
        float or None: The average orthogonality loss, or None if no SORSA matrices are found.
    """
    ortho_loss = 0.0  # 初始化正交损失为 0.0
    den = 0  # 初始化 SORSA 矩阵的计数器为 0

    # 遍历模型中的所有参数
    for name, param in model.named_parameters():
        # 检查参数名称中是否包含 "fossv_A"，表示这是一个 SORSA 矩阵 A
        if "fossv_A" in name:
            a = param  # 获取当前的 SORSA 矩阵 A
            ia = torch.eye(a.shape[0], device=a.device)  # 创建一个与 A 矩阵行数相同的单位矩阵
            ia.requires_grad = False  # 设置单位矩阵不需要梯度计算
            a = a @ a.T - ia  # 计算 A @ A.T - I，如果 A 是正交矩阵，结果应接近零矩阵
            ortho_loss += torch.norm(a, p="fro")  # 计算 Frobenius 范数并累加到正交损失中
            den += 1  # 增加 SORSA 矩阵的计数

        # 检查参数名称中是否包含 "fossv_B"，表示这是一个 SORSA 矩阵 B
        elif "fossv_B" in name:
            b = param  # 获取当前的 SORSA 矩阵 B
            ib = torch.eye(b.shape[1], device=b.device)  # 创建一个与 B 矩阵列数相同的单位矩阵
            ib.requires_grad = False  # 设置单位矩阵不需要梯度计算
            b = b.T @ b - ib  # 计算 B.T @ B - I，如果 B 是正交矩阵，结果应接近零矩阵
            ortho_loss += torch.norm(b, p="fro")  # 计算 Frobenius 范数并累加到正交损失中
            den += 1  # 增加 SORSA 矩阵的计数

    # 如果至少找到一个 SORSA 矩阵，返回平均正交损失
    if den != 0:
        return ortho_loss / den
    else:
        return None  # 如果没有找到任何 SORSA 矩阵，返回 None
