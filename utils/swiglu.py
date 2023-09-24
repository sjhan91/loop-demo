from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """
    This is the implementation of FFN-SwiGLU
    Refer to https://github.com/facebookresearch/xformers/blob/0e520d20f6b570cdaf02a46ed6b27ab3155eef1d/xformers/ops/swiglu_op.py#L390
    """

    def __init__(self, in_feat, hidden_feat):
        super().__init__()

        self.w12 = nn.Linear(in_feat, 2 * hidden_feat)
        self.w3 = nn.Linear(hidden_feat, in_feat)

    def forward(self, x):
        x, gate = self.w12(x).chunk(2, dim=-1)
        x = self.w3(x * F.silu(gate))

        return x
