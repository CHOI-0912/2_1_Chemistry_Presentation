"""v2/fulltrain_best.pt(SimpleModel)의 가중치를 브라우저 JS용으로 export.

두 파일을 만든다:
  weights_simple.json — config + 텐서 목록(이름/shape/offset/size) + Etot_table
  weights_simple.bin  — Etot_table을 뺀 모든 파라미터를 float32 리틀엔디안으로 이어붙인 블롭

80만 파라미터를 JSON 숫자 배열로 쓰면 13MB가 넘고(파싱도 느림) base64로 감싸도 4.3MB라,
그냥 raw float32 블롭으로 뺐다. JS는 fetch().arrayBuffer() → Float32Array 로 바로 읽으므로
디코더 라이브러리가 필요 없다. Etot_table만은 값이 수백 Ha라 float32(유효 ~7자리)면
원자당 1e-4 Ha가 뭉개지므로 JSON에 float64 그대로 둔다.

    python v2/simulator/export_simple_weights.py
"""

import json
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "v2"))

CKPT = ROOT / "v2" / "fulltrain_best.pt"
OUT_DIR = Path(__file__).resolve().parent

# fulltrain_best.pt 학습 설정. 체크포인트 shape과 대조해 검증한다(아래 EXPECT).
CONFIG = dict(num_atom_whole=92, atten_heads=16, atten_dim=256, inner_dim=128, number_propo=2)

sd = torch.load(CKPT, map_location="cpu", weights_only=True)

# --- config 검증: shape이 하나라도 어긋나면 JS가 조용히 틀린 값을 낸다 ---
heads, adim, idim, npro = (CONFIG["atten_heads"], CONFIG["atten_dim"],
                           CONFIG["inner_dim"], CONFIG["number_propo"])
head_qk, head_v, ffn = adim // heads, idim // heads, idim * 4
EXPECT = {
    "log_self_distance": (),
    "A1_WQK.weight": (npro * head_qk * 2, 1), "A1_WQK.bias": (npro * head_qk * 2,),
    "A1_WV.weight": (npro * head_v, 1), "A1_WV.bias": (npro * head_v,),
    "embq.weight": (CONFIG["num_atom_whole"] + 1, adim - npro * head_qk),
    "embk.weight": (CONFIG["num_atom_whole"] + 1, adim - npro * head_qk),
    "embv.weight": (CONFIG["num_atom_whole"] + 1, idim - npro * head_v),
    "energy_head.weight": (1, idim), "energy_head.bias": (1,),
    "Etot_table": (CONFIG["num_atom_whole"] + 1,),
}
for blk in ("A1", "A2", "A3"):
    EXPECT[f"{blk}_SwiGLUFFN.w_gate.weight"] = (ffn, idim)
    EXPECT[f"{blk}_SwiGLUFFN.w_up.weight"] = (ffn, idim)
    EXPECT[f"{blk}_SwiGLUFFN.w_down.weight"] = (idim, ffn)
for blk in ("A2", "A3"):
    EXPECT[f"{blk}_WQK.weight"] = (adim * 2, idim)
    EXPECT[f"{blk}_WQK.bias"] = (adim * 2,)
    EXPECT[f"{blk}_WV.weight"] = (idim, idim)
    EXPECT[f"{blk}_WV.bias"] = (idim,)

missing = set(EXPECT) - set(sd)
extra = set(sd) - set(EXPECT)
if missing or extra:
    raise SystemExit(f"state_dict 키 불일치: missing={sorted(missing)} extra={sorted(extra)}")
for name, shape in EXPECT.items():
    got = tuple(sd[name].shape)
    if got != shape:
        raise SystemExit(f"{name} shape {got} != 기대 {shape} — CONFIG가 체크포인트와 다름")

# --- 블롭 쓰기 (Etot_table 제외, state_dict 순서 그대로) ---
blob = bytearray()
tensors = []
for name, t in sd.items():
    if name == "Etot_table":
        continue
    flat = t.detach().float().cpu().contiguous().view(-1)
    tensors.append({"name": name, "shape": list(t.shape),
                    "offset": len(blob) // 4, "size": flat.numel()})
    blob += struct.pack(f"<{flat.numel()}f", *flat.tolist())

manifest = {
    "model": "SimpleModel",
    "source": "v2/fulltrain_best.pt",
    "config": {**CONFIG, "head_qk_dim": head_qk, "head_v_dim": head_v, "ffn_dim": ffn},
    "dtype": "float32",
    "count": len(blob) // 4,
    "tensors": tensors,
    # 고립원자 총에너지(Hartree). 인덱스 = 원자번호, [0]은 패딩용 0.
    "Etot_table": sd["Etot_table"].double().tolist(),
}

bin_path = OUT_DIR / "weights_simple.bin"
json_path = OUT_DIR / "weights_simple.json"
bin_path.write_bytes(bytes(blob))
json_path.write_text(json.dumps(manifest), encoding="utf-8")

print(f"wrote {bin_path} ({bin_path.stat().st_size:,} bytes, {manifest['count']:,} float32)")
print(f"wrote {json_path} ({json_path.stat().st_size:,} bytes)")
