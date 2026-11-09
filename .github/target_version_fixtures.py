"""Generate exact-version models and check a native pnnx binary against them."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
import torchvision


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RoiAlignModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roi_align = torchvision.ops.RoIAlign((3, 3), 0.25, 3, False)
        self.register_buffer("rois", torch.tensor([[0, 0, 10, 12, 20]], dtype=torch.float))

    def forward(self, x):
        return self.roi_align(x, self.rois)


def generate(directory):
    assert torch.__version__.split("+")[0] == "2.15.0"
    assert torchvision.__version__.split("+")[0] == "0.29.2"
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    x = torch.rand(1, 12, 64, 64)
    model = RoiAlignModel().eval()
    with torch.no_grad():
        expected = model(x)
        traced = torch.jit.trace(model, x)
    assert "torchvision::roi_align" in str(traced.inlined_graph)
    traced.save(str(directory / "vision_roi_align.pt"))
    np.save(directory / "input.npy", x.numpy())
    np.save(directory / "expected.npy", expected.numpy())
    (directory / "graph.txt").write_text(str(traced.inlined_graph))
    manifest = {
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "shape": list(x.shape),
        "sha256": {p.name: sha256(p) for p in directory.iterdir() if p.name != "manifest.json"},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def verify(directory, binary, work):
    binary = binary.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["torch"].split("+")[0] == "2.15.0"
    assert manifest["torchvision"].split("+")[0] == "0.29.2"
    for name, expected in manifest["sha256"].items():
        assert sha256(directory / name) == expected, name
    print("Fixture producer:", manifest["torch"], manifest["torchvision"])
    print("Python verifier:", torch.__version__, torchvision.__version__, np.__version__)
    print("Native binary:", binary, "SHA256:", sha256(binary))
    work.mkdir(parents=True, exist_ok=True)
    for name in manifest["sha256"]:
        shutil.copy2(directory / name, work / name)
    os.chdir(work)
    subprocess.run([str(binary), "vision_roi_align.pt", "inputshape=[1,12,64,64]", "fp16=0"], check=True)
    for suffix in (".pnnx.param", ".pnnx.bin", "_pnnx.py"):
        assert Path("vision_roi_align" + suffix).stat().st_size > 0
    assert "torchvision.ops.RoIAlign" in Path("vision_roi_align.pnnx.param").read_text()
    spec = importlib.util.spec_from_file_location("vision_roi_align_pnnx", "vision_roi_align_pnnx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The older verifier never loads the new TorchScript file. Only the native
    # 2.15-linked converter reads it; Python executes the generated model.
    with torch.no_grad():
        actual = module.Model().eval()(torch.from_numpy(np.load("input.npy", allow_pickle=False)))
    expected = torch.from_numpy(np.load("expected.npy", allow_pickle=False))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    print("PASS: native pnnx converted a 2.15.0/0.29.2 ROIAlign fixture with matching output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "verify"))
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    if args.mode == "generate":
        generate(args.fixtures.resolve())
    else:
        if args.binary is None or args.work is None:
            parser.error("verify requires --binary and --work")
        verify(args.fixtures.resolve(), args.binary, args.work.resolve())
