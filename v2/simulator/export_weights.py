"""v2/nnp.pt(SimpleNNP)의 가중치를 브라우저 JS용 weights.json으로 export.

nnp.pt를 다시 학습(`python v2/train.py`)했으면 이 스크립트로 weights.json도 갱신한다.
state_dict를 그대로 JSON 배열로 옮길 뿐이다.

    python v2/simulator/export_weights.py
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "v2"))
from model import SimpleNNP, MAX_Z  # noqa: E402

HID, NRBF, CUT, NBLK = 64, 32, 5.0, 3  # model.py 기본값과 동일해야 함

model = SimpleNNP(hidden=HID, n_rbf=NRBF, cutoff=CUT, n_blocks=NBLK)
model.load_state_dict(torch.load(ROOT / "v2" / "nnp.pt", map_location="cpu"))
model.eval()

sd = model.state_dict()
arr = lambda name: sd[name].cpu().numpy().tolist()  # noqa: E731

weights = {
    "config": {"hidden": HID, "n_rbf": NRBF, "cutoff": CUT, "n_blocks": NBLK,
               "gamma": (NRBF / CUT) ** 2, "max_z": MAX_Z},
    "centers": arr("centers"),
    "embedding": arr("embedding.weight"),
    "filters": [
        {"w0": arr(f"filters.{b}.0.weight"), "b0": arr(f"filters.{b}.0.bias"),
         "w2": arr(f"filters.{b}.2.weight"), "b2": arr(f"filters.{b}.2.bias")}
        for b in range(NBLK)
    ],
    "updates": [
        {"w0": arr(f"updates.{b}.0.weight"), "b0": arr(f"updates.{b}.0.bias"),
         "w2": arr(f"updates.{b}.2.weight"), "b2": arr(f"updates.{b}.2.bias")}
        for b in range(NBLK)
    ],
    "readout": {"w0": arr("readout.0.weight"), "b0": arr("readout.0.bias"),
                "w2": arr("readout.2.weight"), "b2": arr("readout.2.bias")},
}

out = Path(__file__).resolve().parent / "weights.json"
out.write_text(json.dumps(weights), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes)")
