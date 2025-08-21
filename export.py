import torch

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from brevitas.export import export_qonnx
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.core.onnx_exec import execute_onnx

from models.yolo import get_model



model_path = "runs/train/quantyolov8_8w8a_416/weights/best.pt"
cfg = "runs/train/quantyolov8_8w8a_416/cfg.yaml"

model, _, _ = get_model(cfg, model_path, 1, "cpu", load_ema=True)
model.eval()

test_input = torch.zeros(1, 3, 416, 416).to("cpu").type_as(next(model.parameters()))
input_dict = {"global_in": test_input.numpy().astype(np.float32) / 255.0}

def execute_as_onnx(model, onnx_name):
    qonnx_model = export_qonnx(model, test_input, onnx_name)
    qonnx_model = ModelWrapper(onnx_name)
    qonnx_model = qonnx_model.transform(InferShapes())
    qonnx_model = qonnx_model.transform(GiveUniqueNodeNames())
    qonnx_model = qonnx_model.transform(GiveReadableTensorNames())
    qonnx_model.save(onnx_name)
    output_dict = execute_onnx(qonnx_model, input_dict, return_full_exec_context=True)
    return output_dict


output_brevitas = model(test_input)
output_dict = execute_as_onnx(model, 'model_{}.onnx'.format("cpu"))