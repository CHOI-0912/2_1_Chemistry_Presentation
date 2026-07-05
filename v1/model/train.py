# train.py — ANI-1x ccsd(t)_cbs 총에너지로 Chemical_Model 학습
#
# 기존 코드 활용:
#   - preprocess.iter_molecules() : 분자식 그룹을 lazy하게 읽음 (NaN 에너지는 제외)
#   - model.Chemical_Model        : [b,n] 종 인덱스 + [b,n,3] 좌표 -> [b] 총에너지(extensive)
#
# 핵심 설계:
#   - 모델에 패딩/마스킹이 없어 한 배치의 원자 수 n이 같아야 함 -> 원자 수(Na)별로 배칭.
#     (Na가 같으면 분자식이 달라도 같은 배치에 섞어도 됨 — 종은 임베딩으로 구분되므로.)
#   - 타깃 정규화: 원소(H/C/N/O)별 기준에너지를 최소제곱으로 구해 차감(원자화에너지 유사),
#                 남은 잔차를 std로 스케일해서 학습. 평가 때 역변환해 Hartree MAE 보고.
#   - conformer의 약 10%를 검증셋으로 분리, 매 에폭 검증 MAE 출력, best/last 가중치 저장.

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from preprocess import iter_molecules
from model import Chemical_Model

# --- 하이퍼파라미터 (필요시 조정) ---
DIM = 64
REL_DIM = 64
NUM_SPECIES = 4          # H, C, N, O
EPOCHS = 30
BATCH = 64
LR = 1e-3
VAL_RATIO = 0.1
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
HARTREE2KCAL = 627.5094740631


def load_grouped():
    """iter_molecules로 valid(non-NaN) conformer를 읽어 원자 수(Na)별로 묶는다."""
    by_na = {}
    for mol in iter_molecules():
        e = mol["energy"]
        valid = ~np.isnan(e)
        if not valid.any():
            continue
        z = mol["atomic_numbers"].astype(np.int64)          # (Na,)
        xyz = mol["coordinates"][valid].astype(np.float32)  # (k, Na, 3)
        ev = e[valid].astype(np.float64)                    # (k,)
        Na = int(z.shape[0])
        d = by_na.setdefault(Na, {"z": [], "xyz": [], "e": []})
        d["z"].append(np.broadcast_to(z, (ev.shape[0], Na)).copy())
        d["xyz"].append(xyz)
        d["e"].append(ev)
    for d in by_na.values():
        d["z"] = np.concatenate(d["z"])      # (M, Na)
        d["xyz"] = np.concatenate(d["xyz"])  # (M, Na, 3)
        d["e"] = np.concatenate(d["e"])      # (M,)
    return by_na


def species_counts(z):
    """(M, Na) 종 인덱스 -> (M, NUM_SPECIES) 원소별 개수."""
    return np.stack([(z == s).sum(1) for s in range(NUM_SPECIES)], axis=1).astype(np.float64)


def save_ckpt(path, model, ref_coef, target_std, epoch, val_mae):
    """가중치 + 역정규화에 필요한 상수(원소 기준에너지, 잔차 std)를 함께 저장."""
    torch.save({
        "model_state": model.state_dict(),
        "config": {"dim": DIM, "rel_dim": REL_DIM, "num_atoms": NUM_SPECIES},
        "energy_ref": np.asarray(ref_coef, dtype=np.float64),  # (4,) H,C,N,O Hartree
        "target_std": float(target_std),
        "epoch": epoch,
        "val_mae_hartree": float(val_mae),
    }, path)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    by_na = load_grouped()
    total = sum(d["e"].shape[0] for d in by_na.values())
    print(f"불러온 conformer 수: {total:,} (원자 수 그룹 {len(by_na)}개), device={DEVICE}")

    # 1) Na별 train/val 인덱스 분리
    rng = np.random.default_rng(SEED)
    splits = {}
    train_counts, train_e = [], []
    for Na, d in by_na.items():
        M = d["e"].shape[0]
        idx = rng.permutation(M)
        n_val = max(1, round(M * VAL_RATIO)) if M > 1 else 0
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        splits[Na] = (tr_idx, val_idx)
        train_counts.append(species_counts(d["z"][tr_idx]))
        train_e.append(d["e"][tr_idx])

    # 2) 원소별 기준에너지 최소제곱 적합 (train만 사용 — 누수 방지)
    C = np.concatenate(train_counts)   # (Ntr, 4)
    E = np.concatenate(train_e)        # (Ntr,)
    ref_coef, *_ = np.linalg.lstsq(C, E, rcond=None)   # (4,)
    target_std = float((E - C @ ref_coef).std())
    print(f"원소별 기준에너지 (H,C,N,O) Hartree: {np.round(ref_coef, 4)}")
    print(f"잔차 std (Hartree): {target_std:.4f}")

    # 3) Na별 TensorDataset / DataLoader  (타깃 = (e - counts@ref)/std)
    train_loaders, val_loaders = [], []
    for Na, d in by_na.items():
        tr_idx, val_idx = splits[Na]
        target = (d["e"] - species_counts(d["z"]) @ ref_coef) / target_std
        z_t = torch.from_numpy(d["z"]).long()
        xyz_t = torch.from_numpy(d["xyz"]).float()
        tgt_t = torch.from_numpy(target).float()
        if len(tr_idx):
            train_loaders.append(DataLoader(
                TensorDataset(z_t[tr_idx], xyz_t[tr_idx], tgt_t[tr_idx]),
                batch_size=BATCH, shuffle=True))
        if len(val_idx):
            val_loaders.append(DataLoader(
                TensorDataset(z_t[val_idx], xyz_t[val_idx], tgt_t[val_idx]),
                batch_size=BATCH, shuffle=False))

    # 4) 모델 / 옵티마이저
    model = Chemical_Model(dim=DIM, rel_dim=REL_DIM, num_atoms=NUM_SPECIES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    ref_t = torch.tensor(ref_coef, dtype=torch.float32, device=DEVICE)

    best_val = float("inf")
    val_mae = float("nan")
    for epoch in range(1, EPOCHS + 1):
        # --- train: 모든 Na 그룹의 배치를 모아 섞어서 학습 ---
        model.train()
        batches = []
        for ld in train_loaders:
            batches.extend(ld)          # shuffle=True라 매번 재셔플됨
        random.shuffle(batches)         # Na 그룹 간에도 섞기
        tr_loss = tr_n = 0
        for z, xyz, tgt in batches:
            z, xyz, tgt = z.to(DEVICE), xyz.to(DEVICE), tgt.to(DEVICE)
            pred = model(z, xyz)
            loss = loss_fn(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * z.size(0)
            tr_n += z.size(0)

        # --- val: 역정규화해서 실제 총에너지 MAE(Hartree) 평가 ---
        model.eval()
        abs_err = v_n = 0.0
        with torch.no_grad():
            for ld in val_loaders:
                for z, xyz, tgt in ld:
                    z, xyz, tgt = z.to(DEVICE), xyz.to(DEVICE), tgt.to(DEVICE)
                    out = model(z, xyz)
                    counts = torch.stack([(z == s).sum(1) for s in range(NUM_SPECIES)], 1).float()
                    ref = counts @ ref_t
                    pred_E = out * target_std + ref      # 예측 총에너지
                    true_E = tgt * target_std + ref      # 실제 총에너지 (= 원래 e)
                    abs_err += (pred_E - true_E).abs().sum().item()
                    v_n += z.size(0)
        val_mae = abs_err / max(1, v_n)
        print(f"[{epoch:3d}/{EPOCHS}] train MSE(norm)={tr_loss / max(1, tr_n):.4f}  "
              f"val MAE={val_mae:.5f} Ha ({val_mae * HARTREE2KCAL:.2f} kcal/mol)")

        # --- best 가중치 저장 ---
        if val_mae < best_val:
            best_val = val_mae
            save_ckpt(os.path.join(CKPT_DIR, "checkpoint_best.pt"),
                      model, ref_coef, target_std, epoch, val_mae)

    # --- 마지막 가중치 저장 ---
    save_ckpt(os.path.join(CKPT_DIR, "checkpoint_last.pt"),
              model, ref_coef, target_std, EPOCHS, val_mae)
    print(f"학습 완료. best val MAE = {best_val:.5f} Ha -> checkpoints/checkpoint_best.pt")


if __name__ == "__main__":
    main()
