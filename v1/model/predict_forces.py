# predict_forces.py — 학습된 에너지 모델로 per-atom 힘 F = -∂E/∂r 추론 (autograd)
#
# 원리(MLIP/NNP 표준): 힘 = 에너지의 위치에 대한 미분의 음수. F_i = -∂E/∂r_i.
#   에너지 모델이 미분가능하므로 autograd로 ∂E/∂r 를 한 번에 계산한다.
#   학습 시 타깃은 E = out*std + counts@E_ref 로 역정규화되는데,
#   counts@E_ref(원소 기준에너지)는 좌표와 무관한 상수라 힘에 기여하지 않음:
#       F = -∂E/∂r = -std * ∂(out)/∂r        [Hartree/Å]
#
# 입력 스키마(학습 데이터와 동일 계열):
#   Z : 원자번호 (1=H, 6=C, 7=N, 8=O).  shape (n,) 또는 (b, n)
#   R : 좌표(Å).                         shape (n, 3) 또는 (b, n, 3)
# 출력:
#   {"energy_hartree": (b,) 총에너지, "forces": (b, n, 3) Hartree/Å}
#   (단일 분자 입력이면 배치 차원 없이 반환)

import os
import numpy as np
import torch

from preprocess import to_species_index
from model import Chemical_Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CKPT = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
LEGACY_CKPT = os.path.join(BASE_DIR, "checkpoint_best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HARTREE2KCAL = 627.5094740631
NUM_SPECIES = 4


def load_model(ckpt_path=CKPT):
    """train.py가 저장한 체크포인트에서 모델 + 역정규화 상수(E_ref, std)를 복원."""
    if not os.path.exists(ckpt_path) and ckpt_path == CKPT and os.path.exists(LEGACY_CKPT):
        ckpt_path = LEGACY_CKPT
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"체크포인트가 없음: {ckpt_path}\n"
            f"먼저 train.py로 학습해 checkpoints/checkpoint_best.pt를 만드세요.")
    # weights_only=False: 우리가 직접 저장한 체크포인트(신뢰 가능)이고 energy_ref가 numpy 배열이라 필요
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = Chemical_Model(**ck["config"]).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    ref = torch.tensor(ck["energy_ref"], dtype=torch.float32, device=DEVICE)  # (4,)
    std = float(ck["target_std"])
    return model, ref, std


def predict_energy_forces(Z, R, ckpt=None):
    """원자번호 Z, 좌표 R(Å)로 총에너지(Hartree)와 per-atom 힘(Hartree/Å)을 계산.

    ckpt: (model, ref, std) 튜플을 넘기면 그걸 사용(반복 호출 시 재로딩 방지). 없으면 디스크에서 로드.
    """
    model, ref, std = ckpt if ckpt is not None else load_model()

    Z = np.asarray(Z)
    R = np.asarray(R, dtype=np.float32)
    single = (Z.ndim == 1)
    if single:
        Z, R = Z[None], R[None]                      # (1,n), (1,n,3)

    species = to_species_index(Z).astype(np.int64)   # 1,6,7,8 -> 0..3
    z = torch.from_numpy(species).long().to(DEVICE)
    coords = torch.tensor(R, dtype=torch.float32, device=DEVICE, requires_grad=True)

    out = model(z, coords)                           # (b,) 정규화 에너지
    # 분자끼리 독립이라 배치 합의 grad가 각 분자의 ∂out/∂r 를 그대로 보존
    grad = torch.autograd.grad(out.sum(), coords)[0] # (b,n,3) = ∂out/∂r
    forces = (-std * grad).detach().cpu().numpy()    # F = -∂E/∂r [Hartree/Å]

    counts = torch.stack([(z == s).sum(1) for s in range(NUM_SPECIES)], 1).float()
    energy = (out.detach() * std + counts @ ref).cpu().numpy()  # (b,) Hartree

    if single:
        return {"energy_hartree": float(energy[0]), "forces": forces[0]}
    return {"energy_hartree": energy, "forces": forces}


if __name__ == "__main__":
    # 데모: 물 분자(H2O) 한 개. 원자번호 O,H,H + 대략적 좌표(Å).
    Z = np.array([8, 1, 1])
    R = np.array([[0.000,  0.000, 0.119],
                  [0.000,  0.763, -0.477],
                  [0.000, -0.763, -0.477]], dtype=np.float32)

    res = predict_energy_forces(Z, R)
    f = res["forces"]
    print(f"총에너지: {res['energy_hartree']:.6f} Hartree "
          f"({res['energy_hartree'] * HARTREE2KCAL:.2f} kcal/mol)")
    print("per-atom 힘 (Hartree/Å):")
    for sym, fi in zip(["O", "H", "H"], f):
        print(f"  {sym}: [{fi[0]:+.5f}, {fi[1]:+.5f}, {fi[2]:+.5f}]")
    print(f"알짜 힘 합(병진 불변이면 ≈0): {f.sum(0)}")
