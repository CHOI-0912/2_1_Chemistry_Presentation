"""dataloader 사용 예시 — QM9x 일부로 SimpleModel을 잠깐 학습시킨다.

성능을 내려는 스크립트가 아니라, dataloader가 내주는 (이름, 좌표, 원자번호, 에너지, 힘)을
그대로 받아 모델에 먹이는 최소 예시다.

    python v2/train.py
"""

import sys

import numpy as np
import torch
import torch.nn as nn

from dataloader import load_qm9x
from model_Simple import SimpleModel

N_MOL = 5000  # 데이터셋 전체(13만) 중 앞의 일부만
EPOCHS = 5
BATCH = 32
LR = 1e-3
HARTREE2KCAL = 627.5094740631

def load_dataset(n_mol):
    """dataloader를 돌려 패딩된 배열로 쌓는다. 분자마다 원자 수가 달라 0으로 패딩."""
    samples = []
    for i, sample in enumerate(load_qm9x()):
        if i >= n_mol:
            break
        samples.append(sample)

    n_max = max(len(z) for _, _, z, _, _ in samples)
    Z = np.zeros((len(samples), n_max), dtype=np.int64)
    R = np.zeros((len(samples), n_max, 3), dtype=np.float32)
    F = np.zeros((len(samples), n_max, 3), dtype=np.float32)
    E = np.zeros(len(samples), dtype=np.float32)

    for i, (_name, coords, numbers, energy, forces) in enumerate(samples):
        n = len(numbers)
        Z[i, :n] = numbers
        R[i, :n] = coords
        F[i, :n] = forces
        E[i] = energy

    print(f"QM9x {len(samples)} 구조, 최대 {n_max} 원자")
    return Z, R, E, F


def mae(model, Z, R, E, F, idx):
    """에너지/힘 MAE (kcal/mol, kcal/mol/Å)"""
    e_sum = f_sum = n_comp = 0.0
    for b in np.array_split(idx, max(1, len(idx) // BATCH)):
        z, r = torch.from_numpy(Z[b]), torch.from_numpy(R[b])
        e_pred, f_pred = model.energy_and_forces(z, r)
        mask = (z > 0).float()
        e_sum += (e_pred - torch.from_numpy(E[b])).abs().sum().item()
        f_sum += ((f_pred - torch.from_numpy(F[b])).abs().sum(-1) * mask).sum().item()
        n_comp += mask.sum().item() * 3
    return e_sum / len(idx) * HARTREE2KCAL, f_sum / n_comp * HARTREE2KCAL


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    Z, R, E, F = load_dataset(N_MOL)
    perm = rng.permutation(len(E))
    val_idx, train_idx = perm[:500], perm[500:]

    model = SimpleModel(emb_dim=64, num_atom_whole=92, atten_heads=4, atten_dim=64, inner_dim=64, number_propo=2)
    print(f"파라미터 {sum(p.numel() for p in model.parameters())}개")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        for b in np.array_split(rng.permutation(train_idx), len(train_idx) // BATCH):
            z, r = torch.from_numpy(Z[b]), torch.from_numpy(R[b])
            e_pred, f_pred = model.energy_and_forces(z, r, create_graph=True)

            mask = (z > 0).float()[..., None]
            loss_e = mse(e_pred, torch.from_numpy(E[b]))
            loss_f = (((f_pred - torch.from_numpy(F[b])) * mask) ** 2).sum() / mask.sum() / 3
            loss = loss_e + loss_f

            opt.zero_grad()
            loss.backward()
            opt.step()

        e_mae, f_mae = mae(model, Z, R, E, F, val_idx)
        print(
            f"epoch {epoch}/{EPOCHS}  val E MAE {e_mae:7.2f} kcal/mol  "
            f"F MAE {f_mae:5.2f} kcal/mol/Å",
            flush=True,
        )

    torch.save(model.state_dict(), "v2/model_simple.pt")  # nnp.pt(구 SimpleNNP, 시뮬레이터용)와 분리
    print("v2/model_simple.pt 저장")


if __name__ == "__main__":
    main()
