"""W&B 스윕/단독 실행용 학습 스크립트 — QM9x, 조성 분할(splits.py) 기준.

단독:  python v2/train_wandb.py [--lr 1e-3 --epochs 10 ...]
스윕:  wandb sweep v2/sweep.yaml && wandb agent <sweep-id>
로그인 없이 테스트: WANDB_MODE=offline

프로젝트명: 2_1chemical.
불가능 조합(배수/propo 제약)과 NaN 발산 런은 최악 점수(1e9)를 기록하고 즉시 종료해
Bayes 탐색이 그 영역을 피하게 한다 (크래시로 죽이면 아무것도 못 배움).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_Simple import SimpleModel
from splits import load_split

HARTREE2KCAL = 627.5094740631
WORST = 1e9  # 실패 런에 기록하는 최악 점수

DEFAULTS = dict(
    lr=1e-3, batch=64, epochs=10,
    atten_dim=64, inner_dim=64, atten_heads=4, number_propo=2,
    lambda_F=1.0,
    n_train=50_000,  # Stage 1은 train 앞부분만. 0이면 전부
    seed=0,
)


def bail(reason):
    print("BAIL:", reason, flush=True)
    wandb.log({"val/E_MAE_kcal": WORST, "val/F_MAE_kcal": WORST})
    wandb.finish()
    sys.exit(0)


def load_padded(split, limit=0):
    """load_split(qm9x)을 돌려 패딩 배열로 쌓는다. 최초 1회만 h5를 순회하고 npz 캐시."""
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"cache_qm9x_{split}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        Z, R, E, F = d["Z"], d["R"], d["E"], d["F"]
    else:
        samples = list(load_split("qm9x", split))
        n_max = max(len(numbers) for _, _, numbers, _, _ in samples)
        Z = np.zeros((len(samples), n_max), dtype=np.int64)
        R = np.zeros((len(samples), n_max, 3), dtype=np.float32)
        F = np.zeros((len(samples), n_max, 3), dtype=np.float32)
        E = np.zeros(len(samples), dtype=np.float64)  # 총에너지(수백 Ha)는 float64 유지
        for i, (_name, coords, numbers, energy, forces) in enumerate(samples):
            n = len(numbers)
            Z[i, :n] = numbers
            R[i, :n] = coords
            F[i, :n] = forces
            E[i] = energy
        np.savez_compressed(cache, Z=Z, R=R, E=E, F=F)
        print(f"{split} 캐시 생성: {cache}", flush=True)
    if limit and limit < len(E):
        Z, R, E, F = Z[:limit], R[:limit], E[:limit], F[:limit]
    return Z, R, E, F


def evaluate(model, Z, R, E, F, batch, device):
    """val E MAE(kcal/mol), F MAE(kcal/mol/Å). 힘에 1차 미분이 필요해 no_grad는 못 쓴다."""
    e_sum = f_sum = f_cnt = 0.0
    for s in range(0, len(E), batch):
        z = torch.from_numpy(Z[s:s + batch]).to(device)
        r = torch.from_numpy(R[s:s + batch]).to(device)
        e_pred, f_pred = model.energy_and_forces(z, r)
        mask = (z > 0).float()
        e_sum += np.abs(e_pred.detach().cpu().numpy() - E[s:s + batch]).sum()
        f_sum += ((f_pred.detach().cpu() - torch.from_numpy(F[s:s + batch])).abs().sum(-1) * mask.cpu()).sum().item()
        f_cnt += mask.sum().item() * 3
    return e_sum / len(E) * HARTREE2KCAL, f_sum / f_cnt * HARTREE2KCAL


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔(cp949)에서 Å 등 출력 보호
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    args, _ = p.parse_known_args()

    run = wandb.init(project="2_1chemical", config=vars(args))
    c = wandb.config

    # 불가능 조합 방어 — sweep.yaml 주석 참고
    if c.atten_dim % c.atten_heads or c.inner_dim % c.atten_heads or not (0 < c.number_propo < c.atten_heads):
        bail("불가능 조합 (배수/propo 제약)")

    torch.manual_seed(c.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Ztr, Rtr, Etr, Ftr = load_padded("train", c.n_train)
    Zva, Rva, Eva, Fva = load_padded("val")
    print(f"train {len(Etr)} / val {len(Eva)} / device {device}", flush=True)

    model = SimpleModel(num_atom_whole=92, atten_heads=c.atten_heads, atten_dim=c.atten_dim,
                        inner_dim=c.inner_dim, number_propo=c.number_propo).to(device)
    wandb.summary["n_params"] = sum(p_.numel() for p_ in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=c.lr)
    rng = np.random.default_rng(c.seed)

    best = float("inf")
    for epoch in range(1, c.epochs + 1):
        t0 = time.time()
        perm = rng.permutation(len(Etr))
        for s in range(0, len(perm), c.batch):
            idx = perm[s:s + c.batch]
            z = torch.from_numpy(Ztr[idx]).to(device)
            r = torch.from_numpy(Rtr[idx]).to(device)
            e_t = torch.from_numpy(Etr[idx]).to(device)
            f_t = torch.from_numpy(Ftr[idx]).to(device)

            e, f = model.energy_and_forces(z, r, create_graph=True)
            mask = (z > 0).float()[..., None]
            loss_e = ((e - e_t) ** 2).mean()
            loss_f = (((f - f_t) * mask) ** 2).sum() / mask.sum() / 3
            loss = loss_e + c.lambda_F * loss_f
            if not torch.isfinite(loss):
                bail(f"NaN/Inf loss (epoch {epoch}, step {s // c.batch})")
            opt.zero_grad()
            loss.backward()
            opt.step()

        e_mae, f_mae = evaluate(model, Zva, Rva, Eva, Fva, c.batch, device)
        if not (np.isfinite(e_mae) and np.isfinite(f_mae)):
            bail(f"NaN 평가 (epoch {epoch})")
        wandb.log({
            "epoch": epoch,
            "val/E_MAE_kcal": e_mae, "val/F_MAE_kcal": f_mae,
            "train/loss": loss.item(), "train/loss_E": loss_e.item(), "train/loss_F": loss_f.item(),
            "perf/sec_per_epoch": time.time() - t0,
        })
        print(f"epoch {epoch}/{c.epochs}  val E MAE {e_mae:8.2f} kcal/mol  F MAE {f_mae:7.2f} kcal/mol/Å"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        if e_mae < best:
            best = e_mae
            torch.save(model.state_dict(), os.path.join(run.dir, "best.pt"))

    wandb.summary["best/E_MAE_kcal"] = best
    wandb.finish()


if __name__ == "__main__":
    main()
