"""W&B 스윕/단독 학습 — 전처리된 데이터셋(data/processed/*) 사용. --datasets로 선택.

먼저 `python v2/preprocess.py`로 전처리(ragged npy)가 되어 있어야 한다.
processed에 없는 데이터셋은 건너뛴다(부분 전처리 상태에서도 동작).
기본 데이터셋은 ani1x_ccsdt 제외 4종 — DEFAULTS의 datasets 주석 참고.

단독:  python v2/train_wandb.py [--lr 1e-3 --per_ds 40000 ...]
스윕:  wandb sweep v2/sweep.yaml && wandb agent <sweep-id>
로그인 없이 테스트: WANDB_MODE=offline

프로젝트명: 2_1chemicstary.
- per_ds: 데이터셋별 학습 샘플 상한(0=전부). 스윕은 서브샘플로 빠르게, 본학습은 0으로.
- ani1x_ccsdt는 힘이 없어 힘 손실에서 자동 제외(샘플별 가중 0).
- 불가능 조합·NaN 런은 최악 점수(1e9) 기록 후 즉시 종료 (Bayes가 피하게).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SimpleModel

HARTREE2KCAL = 627.5094740631
WORST = 1e9
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
DS_ALL = ["qm9x", "ani1x_wb97x", "ani1x_ccsdt", "ani2x", "transition1x"]

DEFAULTS = dict(
    lr=1e-3, batch=64, epochs=10,
    atten_dim=64, inner_dim=64, atten_heads=4, number_propo=2,
    lambda_F=100.0,     # MACE 계열 기본값(E:F = 1:100). per-atom 손실로 바꾸며 재조정(2026-08-13)
    clip=1.0,           # 기울기 노름 클리핑 (0=끄기)
    per_ds=40_000,      # 데이터셋별 train 샘플 상한 (0=전부)
    val_per_ds=2_000,   # 데이터셋별 val 샘플 상한
    seed=0,
    # 사용할 데이터셋 (쉼표 구분). 기본값에서 ani1x_ccsdt 제외 (2026-08-13 결정):
    # ccsdt는 wb97x와 같은 좌표에 다른 이론수준(CCSD(T)) 라벨이 붙은 모순 타깃이라
    # (격차 평균 +0.316 Ha, 실측) 혼합 학습에서 MAE 바닥(~127 kcal/mol)을 만들었다.
    datasets="qm9x,ani1x_wb97x,ani2x,transition1x",
    ref="fit",          # 고립원자 몫 기준: "fit"(최소제곱) | "nist"(NIST LSD 고립원자) — 스윕 A/B 축
)


def bail(reason):
    print("BAIL:", reason, flush=True)
    # 데이터셋별 키에도 같이 기록 — 전체 패널에만 1e9가 찍히면 스윕 워크스페이스에서
    # "전체는 치솟는데 데이터셋별은 멀쩡"한 착시를 만든다 (2026-08-10 실사례)
    wandb.log({"val/E_MAE_kcal": WORST, "val/F_MAE_kcal": WORST,
               **{f"val/E_MAE_{ds}": WORST for ds in DS_ALL}})
    wandb.finish()
    sys.exit(0)


def open_split(ds, split):
    d = PROC / ds / split
    if not (d / "energy.npy").exists():
        return None
    return dict(
        name=ds,
        E=np.load(d / "energy.npy", mmap_mode="r"),
        ptr=np.load(d / "ptr.npy"),
        Z=np.load(d / "numbers.npy", mmap_mode="r"),
        R=np.load(d / "coords.npy", mmap_mode="r"),
        F=np.load(d / "forces.npy", mmap_mode="r") if (d / "forces.npy").exists() else None,
    )


def make_index(objs, per_ds, rng):
    """(dataset_idx, mol_idx) 목록. per_ds>0이면 데이터셋별 무작위 서브샘플."""
    idx = []
    for di, o in enumerate(objs):
        M = len(o["E"])
        if 0 < per_ds < M:
            sel = rng.choice(M, per_ds, replace=False)
        else:
            sel = np.arange(M)
        idx.append(np.stack([np.full(len(sel), di), sel], axis=1))
    return np.concatenate(idx)


def collate(objs, items, device):
    """ragged mmap → 배치 내 최대 원자 수로 패딩. fw=힘 타깃 보유 여부(샘플별)."""
    B = len(items)
    ns = [int(objs[di]["ptr"][mi + 1] - objs[di]["ptr"][mi]) for di, mi in items]
    nmax = max(ns)
    Z = np.zeros((B, nmax), np.int64)
    R = np.zeros((B, nmax, 3), np.float32)
    Ft = np.zeros((B, nmax, 3), np.float32)
    fw = np.zeros(B, np.float32)
    E = np.zeros(B, np.float64)
    for b, ((di, mi), n) in enumerate(zip(items, ns)):
        o = objs[di]
        a = int(o["ptr"][mi])
        Z[b, :n] = o["Z"][a:a + n]
        R[b, :n] = o["R"][a:a + n]
        E[b] = o["E"][mi]
        if o["F"] is not None:
            Ft[b, :n] = o["F"][a:a + n]
            fw[b] = 1.0
    return (torch.from_numpy(Z).to(device), torch.from_numpy(R).to(device),
            torch.from_numpy(E).to(device), torch.from_numpy(Ft).to(device),
            torch.from_numpy(fw).to(device))


HARTREE2MEV = 27211.386245988


def evaluate(model, objs, index, batch, device):
    """전체 + 데이터셋별 E MAE(kcal/mol), F MAE(kcal/mol/Å, 성분별),
    그리고 SOTA 비교용 원자당 지표(meV/atom, meV/Å)."""
    e_err = np.zeros(len(objs)); e_cnt = np.zeros(len(objs))
    ea_sum = 0.0  # Σ |ΔE|/N  (원자당 에너지 오차)
    f_sum = f_cnt = 0.0
    for s in range(0, len(index), batch):
        items = index[s:s + batch]
        z, r, e_t, f_t, fw = collate(objs, items, device)
        e, f = model.energy_and_forces(z, r)
        err = (e.detach() - e_t).abs()
        natoms = (z > 0).sum(dim=1).clamp(min=1)
        ea_sum += (err / natoms).sum().item()
        err = err.cpu().numpy()
        for (di, _), v in zip(items, err):
            e_err[di] += v; e_cnt[di] += 1
        mask = (z > 0).float() * fw[:, None]
        f_sum += ((f.detach() - f_t).abs().sum(-1) * mask).sum().item()
        f_cnt += mask.sum().item() * 3
    n = max(1, e_cnt.sum())
    per_ds = {objs[i]["name"]: e_err[i] / max(1, e_cnt[i]) * HARTREE2KCAL for i in range(len(objs))}
    overall = e_err.sum() / n * HARTREE2KCAL
    f_mae = f_sum / max(1, f_cnt) * HARTREE2KCAL
    # SOTA 논문 단위: 에너지 meV/atom(MAE), 힘 meV/Å(성분별 MAE)
    e_mev_atom = ea_sum / n * HARTREE2MEV
    f_mev_ang = f_sum / max(1, f_cnt) * HARTREE2MEV
    return overall, f_mae, per_ds, e_mev_atom, f_mev_ang


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    args, _ = p.parse_known_args()

    run = wandb.init(project="2_1chemicstary", config=vars(args))
    c = wandb.config

    if c.atten_dim % c.atten_heads or c.inner_dim % c.atten_heads or not (0 < c.number_propo < c.atten_heads):
        bail("불가능 조합 (배수/propo 제약)")

    torch.manual_seed(c.seed)
    rng = np.random.default_rng(c.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TF32(암페어+ 행렬곱 가속): 실측 -20% (A10G, 2026-08-17 벤치). torch.compile은 2차 미분
    # 미지원으로 탈락, 버킷 배칭은 TF32 이후 병목이 아니라(커널 오버헤드 지배) 효과 없음.
    torch.set_float32_matmul_precision("high")

    use_ds = [d.strip() for d in c.datasets.split(",") if d.strip()]
    bad = [d for d in use_ds if d not in DS_ALL]
    if bad:
        bail(f"알 수 없는 데이터셋 {bad} (가능: {DS_ALL})")
    tr_objs = [o for ds in use_ds if (o := open_split(ds, "train"))]
    va_objs = [o for ds in use_ds if (o := open_split(ds, "val"))]
    if not tr_objs:
        bail(f"전처리 데이터 없음: {PROC} — python v2/preprocess.py 먼저")
    tr_idx = make_index(tr_objs, c.per_ds, rng)
    va_idx = make_index(va_objs, c.val_per_ds, rng)
    print("train:", {o["name"]: min(len(o["E"]), c.per_ds or len(o["E"])) for o in tr_objs},
          f"= {len(tr_idx):,} / val {len(va_idx):,} / device {device}", flush=True)

    model = SimpleModel(num_atom_whole=92, atten_heads=c.atten_heads, atten_dim=c.atten_dim,
                        inner_dim=c.inner_dim, number_propo=c.number_propo, ref=c.ref).to(device)
    wandb.summary["n_params"] = sum(p_.numel() for p_ in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=c.lr, fused=(device == "cuda"))

    # lr 스케줄: 1에포크 동안 0→lr 선형 워밍업, 이후 남은 에포크에 걸쳐 코사인 감쇠
    # (고정 lr이 후반 발산의 공범이었다 — 2026-08-11 fulltrain에서 ani2x 110→7,648 kcal/mol)
    steps_per_epoch = math.ceil(len(tr_idx) / c.batch)
    warmup_steps = steps_per_epoch
    total_steps = steps_per_epoch * c.epochs

    def lr_scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    best = float("inf")
    for epoch in range(1, c.epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(tr_idx))
        for s in range(0, len(order), c.batch):
            z, r, e_t, f_t, fw = collate(tr_objs, tr_idx[order[s:s + c.batch]], device)
            e, f = model.energy_and_forces(z, r, create_graph=True)
            wm = (z > 0).float()[..., None] * fw[:, None, None]  # 힘 없는 샘플(ccsdt)은 0
            # MACE식 손실(2026-08-13 전환): 에너지를 원자 수로 나눠 제곱 — 큰 분자가 손실을
            # 지배하는 것(예제: 60원자 분자가 물의 400배)을 없애고, √loss_e가 곧 원자당
            # RMSE(Ha/atom)라 SOTA 성적표 단위(meV/atom)와 바로 이어진다.
            natoms = (z > 0).sum(dim=1).clamp(min=1)
            loss_e = (((e - e_t) / natoms) ** 2).mean()
            loss_f = (((f - f_t) * wm) ** 2).sum() / wm.sum().clamp(min=1.0) / 3
            loss = loss_e + c.lambda_F * loss_f
            if not torch.isfinite(loss):
                bail(f"NaN/Inf loss (epoch {epoch}, step {s // c.batch})")
            opt.zero_grad()
            loss.backward()
            # 클리핑: S2·CCl4 같은 소수의 극단 샘플이 배치 기울기를 지배하는 것을 막는다
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip) if c.clip else None
            opt.step()
            sched.step()

        e_mae, f_mae, per_ds, e_mev_atom, f_mev_ang = evaluate(model, va_objs, va_idx, c.batch, device)
        if not (np.isfinite(e_mae) and np.isfinite(f_mae)):
            bail(f"NaN 평가 (epoch {epoch})")
        wandb.log({"epoch": epoch, "val/E_MAE_kcal": e_mae, "val/F_MAE_kcal": f_mae,
                   "val/E_MAE_meV_atom": e_mev_atom, "val/F_MAE_meV_A": f_mev_ang,
                   **{f"val/E_MAE_{k}": v for k, v in per_ds.items()},
                   "train/loss": loss.item(), "train/loss_E": loss_e.item(),
                   "train/loss_F": loss_f.item(), "train/lr": sched.get_last_lr()[0],
                   **({"train/grad_norm": float(gnorm)} if gnorm is not None else {}),
                   "perf/sec_per_epoch": time.time() - t0})
        print(f"epoch {epoch}/{c.epochs}  val E MAE {e_mae:8.2f}  F MAE {f_mae:7.2f} kcal/mol(/Å)  "
              f"[{e_mev_atom:.1f} meV/atom, {f_mev_ang:.1f} meV/Å]  "
              + " ".join(f"{k}={v:.1f}" for k, v in per_ds.items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        if e_mae < best:
            best = e_mae
            torch.save(model.state_dict(), os.path.join(run.dir, "best.pt"))

    wandb.summary["best/E_MAE_kcal"] = best
    wandb.finish()


if __name__ == "__main__":
    main()
