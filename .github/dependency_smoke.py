"""Temporary runtime checks for the dependency upgrade PR."""

import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pnnx
import torch
import torchvision


def convert_model(filename, shape):
    subprocess.run(
        [pnnx.EXEC_PATH, filename, "inputshape=" + shape, "fp16=0"],
        check=True,
    )
    stem = Path(filename).stem
    for suffix in ("_pnnx.py", ".pnnx.param", ".pnnx.bin"):
        assert Path(stem + suffix).stat().st_size > 0
    spec = importlib.util.spec_from_file_location(stem + "_pnnx", stem + "_pnnx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model().eval()


def test_torchvision_roi_align(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.roi_align = torchvision.ops.RoIAlign((3, 3), 0.25, 3, False)
            self.register_buffer("rois", torch.tensor([[0, 0, 10, 12, 20]], dtype=torch.float))

        def forward(self, x):
            return self.roi_align(x, self.rois)

    model = Model().eval()
    x = torch.rand(1, 12, 64, 64)
    with torch.no_grad():
        expected = model(x)
        torch.jit.trace(model, x).save("vision_roi_align.pt")
        actual = convert_model("vision_roi_align.pt", "[1,12,64,64]")(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_onnx_conv_relu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    weight = rng.standard_normal((2, 3, 3, 3)).astype(np.float32)
    bias = rng.standard_normal((2,)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "weight", "bias"], ["conv"]),
         helper.make_node("Relu", ["conv"], ["y"])],
        "dependency-smoke",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 6, 6])],
        [numpy_helper.from_array(weight, "weight"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, "onnx_conv_relu.onnx")
    with torch.no_grad():
        expected = torch.nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(weight), torch.from_numpy(bias)).relu()
        actual = convert_model("onnx_conv_relu.onnx", "[1,3,8,8]")(torch.from_numpy(x))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
