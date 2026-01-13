import numpy as np
import torch
from torch import nn
from copy import copy

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas_examples.imagenet_classification.models.common import CommonIntActQuant, CommonUintActQuant
from brevitas_examples.imagenet_classification.models.common import CommonIntWeightPerChannelQuant
from brevitas.export import export_qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.core.onnx_exec import execute_onnx

from models.yolo import Model, get_model
from models.finn_models import QuantV8Detect, QuantC2f, QuantDetect


class QuantConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            weight_bit_width=4,
            act_bit_width=4,
            padding=None,
            groups=1,
            dilation=1,
            act=True):
        super(QuantConv, self).__init__()
        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.autopad(kernel_size, padding, dilation),
            groups=groups,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-3, momentum=0.03)
        self.act = QuantReLU(
                            act_quant=CommonUintActQuant,
                            bit_width=act_bit_width)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def autopad(self, k, p=None, d=1):  # kernel, padding, dilation
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        return p
    

class QuantBottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, weight_bit_width=4, act_bit_width=4, g=1, k=(3, 3), e=0.5, cv2_act=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = QuantConv(c1, c_, k[0], 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.cv2 = QuantConv(c_, c2, k[1], 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=cv2_act, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class Quantc2f(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, weight_bit_width=4, act_bit_width=4, g=1, e=0.5, act=True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = QuantConv(c1, 2 * self.c, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=True)
        self.cv2 = QuantConv((2 + n) * self.c, c2, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=act)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(QuantBottleneck(self.c, self.c, shortcut=shortcut, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, cv2_act=True, g=g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    

class QuantSPPF(nn.Module):

    def __init__(self, c1, c2, k=5, weight_bit_width=4, act_bit_width=4, act=True):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = QuantConv(c1, c_, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.cv2 = QuantConv(c_ * 4, c2, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
    

class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            QuantConv(3, 16, weight_bit_width=8, act_bit_width=8),
            QuantConv(16, 32),
            Quantc2f(32, 32, 1, True),
            QuantConv(32, 64),
            Quantc2f(64, 64, 2, True),
            QuantConv(64, 128),
            Quantc2f(128, 128, 2, True),
            QuantConv(128, 256),
            Quantc2f(256, 256, 1, True),
            QuantSPPF(256, 256, 5)
        )

    def forward(self, x):
        self.saved_features = []
        for i, m in enumerate(self.model):
            x = m(x)
            self.saved_features.append(x)
        return x
    

def execute_as_onnx(model, onnx_name):
    qonnx_model = export_qonnx(model, test_input, onnx_name)
    qonnx_model = ModelWrapper(onnx_name)
    qonnx_model = qonnx_model.transform(InferShapes())
    qonnx_model = qonnx_model.transform(GiveUniqueNodeNames())
    qonnx_model = qonnx_model.transform(GiveReadableTensorNames())
    qonnx_model.save(onnx_name)
    output_dict = execute_onnx(qonnx_model, input_dict, return_full_exec_context=True)
    return output_dict


def verify_results(brevitas_features, onnx_output_dict):
    testpoints = [
        'Quant_27_out0',
        'Quant_28_out0',
        'Quant_32_out0',
        'Quant_33_out0',
        'Quant_39_out0',
        'Quant_40_out0',
        'Quant_46_out0',
        'Quant_47_out0',
        'Quant_51_out0',
        'global_out'
    ]

    for i, t in enumerate(testpoints):
        feature = brevitas_features[i].detach().cpu().numpy()
        out = onnx_output_dict[t]
        absdiff = np.abs(feature - out)
        mask = absdiff > 0
        numwrong = np.sum(mask)
        print(t, feature.shape, np.sum(mask) / np.product(mask.shape) * 100, "% wrong ({})".format(np.sum(mask)), "mean error:", np.mean(absdiff), "max error:", np.max(absdiff), "mean nonzero:", np.sum(absdiff) / np.sum(mask))
    



device = 'cpu'
weights_A = 'clean_comact.pt'
weights_B = 'clean_indact.pt'
test_input_np = np.load('test_input_192x320.npy')
test_input = torch.from_numpy(test_input_np).float().to(device) / 255.0
input_dict = {"global_in": test_input_np.astype(np.float32) / 255.0}


# instantiate models

model_A = TestModel()
ckpt = torch.load(weights_A)
model_A.load_state_dict(ckpt, strict=True)
model_A.to(device)
model_A.eval()
output_brevitas_A = model_A(test_input)
saved_features_A = [copy(el.detach()) for el in model_A.saved_features]
output_dict_A = execute_as_onnx(model_A, 'model_A_{}.onnx'.format(device))

model_B = TestModel()
ckpt = torch.load(weights_B)
model_B.load_state_dict(ckpt, strict=True)
model_B.to(device)
model_B.eval()
output_brevitas_B = model_B(test_input)
saved_features_B = [copy(el.detach()) for el in model_B.saved_features]
output_dict_B = execute_as_onnx(model_B, 'model_B_{}.onnx'.format(device))

print('MODEL A:')
verify_results(saved_features_A, output_dict_A)

print()
print('MODEL B:')
verify_results(saved_features_B, output_dict_B)
