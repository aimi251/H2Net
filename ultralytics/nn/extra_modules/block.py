import torch
import torch.nn as nn
from collections import OrderedDict

from ..modules.block import get_activation, ConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from .wtconv2d import WTConv2d

__all__ = ['BasicBlock_Ortho', 'BasicBlock_WTConv']

######################################## OrthoNets start ########################################

def gram_schmidt(input):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u
    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x/x.norm(p=2)
        output.append(x)
    return torch.stack(output)

def initialize_orthogonal_filters(c, h, w):

    if h*w < c:
        n = c//(h*w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))

class GramSchmidtTransform(torch.nn.Module):
    instance = {}
    constant_filter: torch.Tensor

    @staticmethod
    def build(c: int, h: int):
        if c not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
        self.register_buffer("constant_filter", rand_ortho_filters.detach())
        
    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W: x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)

class Attention_Ortho(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, FWT: GramSchmidtTransform, input: torch.Tensor):
        #happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input)
        b = input.size(0)
        return input.view(b, -1)

class BasicBlock_Ortho(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', height=64, variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 
        
        self._excitation = nn.Sequential(
            nn.Linear(in_features=ch_out, out_features=round(ch_out / 16), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(ch_out / 16), out_features=ch_out, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention_Ortho()
        self.F_C_A = GramSchmidtTransform.build(ch_out, height)


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        
        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0),out.size(1)
        excitation = self._excitation(compressed).view(b, c, 1, 1)
        out = excitation * out 
        
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)

        return out

class BottleNeck_Ortho(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', height=64, variant='d'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)
        
        self._excitation = nn.Sequential(
            nn.Linear(in_features=ch_out * self.expansion, out_features=round(ch_out / 16 * self.expansion), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(ch_out / 16 * self.expansion), out_features=ch_out * self.expansion, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention_Ortho()
        self.F_C_A = GramSchmidtTransform.build(ch_out * self.expansion, height)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0),out.size(1)
        excitation = self._excitation(compressed).view(b, c, 1, 1)
        out = excitation * out
        
        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out

class Bottleneck_Ortho(Bottleneck):
	def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, height=16):
		super().__init__(c1, c2, shortcut, g, k, e)

		self._excitation = nn.Sequential(
			nn.Linear(in_features=c2, out_features=round(c2 / 16), bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=round(c2 / 16), out_features=c2, bias=False),
			nn.Sigmoid(),
		)
		self.OrthoAttention = Attention_Ortho()
		self.F_C_A = GramSchmidtTransform.build(c2, height)

	def forward(self, x):
		"""'forward()' applies the YOLO FPN to input data."""
		out = self.cv2(self.cv1(x))

		compressed = self.OrthoAttention(self.F_C_A, out)
		b, c = out.size(0),out.size(1)
		excitation = self._excitation(compressed).view(b, c, 1, 1)
		out = excitation * out
		return x + out if self.add else out

class C3_Ortho(C3):
    def __init__(self, c1, c2, n=1, height=16, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_Ortho(c_, c_, shortcut, g, k=(1, 3), e=1.0, height=height) for _ in range(n)))

class C2f_Ortho(C2f):
    def __init__(self, c1, c2, n=1, height=16, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_Ortho(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, height=height) for _ in range(n))

######################################## OrthoNets end ########################################

######################################## Wavelet Convolutions for Large Receptive Fields [ECCV-24] start ########################################

class BasicBlock_WTConv(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = WTConv2d(ch_out, ch_out)

class BottleNeck_WTConv(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = WTConv2d(ch_out, ch_out, stride=stride)

######################################## Wavelet Convolutions for Large Receptive Fields [ECCV-24] end ########################################
