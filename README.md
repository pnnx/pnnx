# pnnx

![download](https://img.shields.io/github/downloads/pnnx/pnnx/total.svg?style=for-the-badge)

PyTorch Neural Network eXchange

Note: The current implementation is in https://github.com/Tencent/ncnn/tree/master/tools/pnnx


## [Download](https://github.com/pnnx/pnnx/releases)

Download PNNX Windows/Linux/MacOS Executable

**https://github.com/pnnx/pnnx/releases**

This package includes all the binaries required. It is portable, so no CUDA or PyTorch runtime environment is needed :)

<table>
<tr>
<td>
  <img src="https://user-images.githubusercontent.com/25181517/186884150-05e9ff6d-340e-4802-9533-2c3f02363ee3.png" width="auto" height="60">
  <br /><b>Windows</b><br />
  <a href="https://github.com/pnnx/pnnx/releases/latest/download/pnnx-20231218-windows.zip">
    <img src="https://img.shields.io/badge/download-blue?style=for-the-badge">
  </a>
</td>
<td>
  <img src="https://github.com/marwin1991/profile-technology-icons/assets/76662862/2481dc48-be6b-4ebb-9e8c-3b957efe69fa" width="auto" height="60">
  <br /><b>Linux</b><br />
  <a href="https://github.com/pnnx/pnnx/releases/latest/download/pnnx-20231218-linux.zip">
    <img src="https://img.shields.io/badge/download-blue?style=for-the-badge">
  </a>
</td>
<td>
  <img src="https://user-images.githubusercontent.com/25181517/186884152-ae609cca-8cf1-4175-8d60-1ce1fa078ca2.png" width="auto" height="60">
  <br /><b>macOS</b><br />
  <a href="https://github.com/pnnx/pnnx/releases/latest/download/pnnx-20231218-macos.zip">
    <img src="https://img.shields.io/badge/download-blue?style=for-the-badge">
  </a>
</td>
<td>
  <img src="https://user-images.githubusercontent.com/25181517/183423507-c056a6f9-1ba8-4312-a350-19bcbc5a8697.png" width="auto" height="60">
  <br /><b>pip install pnnx</b><br />
  <a href="https://pypi.org/project/pnnx">
    <img src="https://img.shields.io/badge/download-blue?style=for-the-badge">
  </a>
</td>
</tr>
</table>

## Usages

1. Export your model to TorchScript

```python
import torch
import torchvision.models as models

net = models.resnet18(pretrained=True)
net = net.eval()

x = torch.rand(1, 3, 224, 224)

# You could try disabling checking when tracing raises error
# mod = torch.jit.trace(net, x, check_trace=False)
mod = torch.jit.trace(net, x)

mod.save("resnet18.pt")
```

2. Convert TorchScript to PNNX

```shell
pnnx resnet18.pt inputshape=[1,3,224,224]
```

Normally, you will get seven files

```resnet18.pnnx.param``` PNNX graph definition

```resnet18.pnnx.bin``` PNNX model weight

```resnet18_pnnx.py``` PyTorch script for inference, the python code for model construction and weight initialization

```resnet18.pnnx.onnx``` PNNX model in onnx format

```resnet18.ncnn.param``` ncnn graph definition

```resnet18.ncnn.bin``` ncnn model weight

```resnet18_ncnn.py``` pyncnn script for inference

3. Visualize PNNX with Netron

Open https://netron.app/ in browser, and drag resnet18.pnnx.param or resnet18.pnnx.onnx into it.

4. PNNX command line options

```
Usage: pnnx [model.pt] [(key=value)...]
  pnnxparam=model.pnnx.param
  pnnxbin=model.pnnx.bin
  pnnxpy=model_pnnx.py
  pnnxonnx=model.pnnx.onnx
  ncnnparam=model.ncnn.param
  ncnnbin=model.ncnn.bin
  ncnnpy=model_ncnn.py
  fp16=1
  optlevel=2
  device=cpu/gpu
  inputshape=[1,3,224,224],...
  inputshape2=[1,3,320,320],...
  customop=/home/nihui/.cache/torch_extensions/fused/fused.so,...
  moduleop=models.common.Focus,models.yolo.Detect,...
Sample usage: pnnx mobilenet_v2.pt inputshape=[1,3,224,224]
              pnnx yolov5s.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320] device=gpu moduleop=models.common.Focus,models.yolo.Detect
```

Parameters:

`pnnxparam` (default="*.pnnx.param", * is the model name): PNNX graph definition file

`pnnxbin` (default="*.pnnx.bin"): PNNX model weight

`pnnxpy` (default="*_pnnx.py"): PyTorch script for inference, including model construction and weight initialization code

`pnnxonnx` (default="*.pnnx.onnx"): PNNX model in onnx format

`ncnnparam` (default="*.ncnn.param"): ncnn graph definition

`ncnnbin` (default="*.ncnn.bin"): ncnn model weight

`ncnnpy` (default="*_ncnn.py"): pyncnn script for inference

`fp16` (default=1): save ncnn weight and onnx in fp16 data type

`optlevel` (default=2): graph optimization level 

| Option | Optimization level              |
|--------|---------------------------------|
|   0    | do not apply optimization       |
|   1    | optimization for inference      |
|   2    | optimization more for inference |

`device` (default="cpu"): device type for the input in TorchScript model, cpu or gpu

`inputshape` (Optional): shapes of model inputs. It is used to resolve tensor shapes in model graph. for example, `[1,3,224,224]` for the model with only 1 input, `[1,3,224,224],[1,3,224,224]` for the model that have 2 inputs.

`inputshape2` (Optional): shapes of alternative model inputs, the format is identical to `inputshape`. Usually, it is used with `inputshape` to resolve dynamic shape (-1) in model graph.

`customop` (Optional): list of Torch extensions (dynamic library) for custom operators, separated by ",". For example, `/home/nihui/.cache/torch_extensions/fused/fused.so,...`

`moduleop` (Optional): list of modules to keep as one big operator, separated by ",". for example, `models.common.Focus,models.yolo.Detect`

## Build from Source

1. Download and setup the libtorch from https://pytorch.org/

2. Clone pnnx (inside Tencent/ncnn tools/pnnx folder)

```shell
git clone https://github.com/Tencent/ncnn.git
```

3. Build with CMake

```shell
mkdir ncnn/tools/pnnx/build
cd ncnn/tools/pnnx/build
cmake -DCMAKE_INSTALL_PREFIX=install -DTorch_INSTALL_DIR=<your libtorch dir> ..
cmake --build . --config Release -j 2
cmake --build . --config Release --target install
```
