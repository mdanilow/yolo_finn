import argparse
import yaml
import torch

from pathlib import Path
from os.path import join

from brevitas.export import export_qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)

from models.yolo import get_model
from models.finn_models import QuantV8Detect, QuantC2f, QuantDetect


parser = argparse.ArgumentParser(prog='test.py')
parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
parser.add_argument('--load_ema', action='store_true')
parser.add_argument('--input_shape', nargs=2, type=int, help='input shape in HW format')
opt = parser.parse_args()

print(opt.input_shape)

with open(opt.data) as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
    nc = int(data['nc'])

model, _, _ = get_model(opt.cfg, opt.weights, nc, load_ema=opt.load_ema)
model = model.eval()

for m in model.modules():
    if isinstance(m, QuantC2f):
        m.forward = m.forward_split
    if isinstance(m, QuantV8Detect) or isinstance(m, QuantDetect):
        m.finn_export = True

build_dir = Path(opt.cfg).parent
exported_filename = str(build_dir / 'exported.onnx')
onnx_model = export_qonnx(model, torch.rand(1, 3, opt.input_shape[0], opt.input_shape[1]), exported_filename)

model = ModelWrapper(exported_filename)
model = model.transform(InferShapes())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = cleanup_model(model)
model.save(exported_filename)
print('Model exported as', exported_filename)