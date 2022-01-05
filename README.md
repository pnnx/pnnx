# pnnx

![download](https://img.shields.io/github/downloads/pnnx/pnnx/total.svg)

PyTorch Neural Network eXchange

Note: The current implementation is in https://github.com/Tencent/ncnn/tree/master/tools/pnnx


## [Download](https://github.com/pnnx/pnnx/releases)

Download PNNX Windows/Linux/MacOS Executable

**https://github.com/pnnx/pnnx/releases**

This package includes all the binaries required. It is portable, so no CUDA or PyTorch runtime environment is needed :)

## Usages

### Example Command

```shell
pnnx.exe mobilenet_v2.pt inputshape=[1,3,224,224]
```

```shell
pnnx.exe yolov5s.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320]
```

### Full Usages

```console
Usage: pnnx [model.pt] [(key=value)...]
  pnnxparam=model.pnnx.param
  pnnxbin=model.pnnx.bin
  pnnxpy=model_pnnx.py
  ncnnparam=model.ncnn.param
  ncnnbin=model.ncnn.bin
  ncnnpy=model_ncnn.py
  optlevel=2
  device=cpu/gpu
  inputshape=[1,3,224,224],...
  inputshape2=[1,3,320,320],...
  customop=/home/nihui/.cache/torch_extensions/fused/fused.so,...
  moduleop=models.common.Focus,models.yolo.Detect,...
```

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
