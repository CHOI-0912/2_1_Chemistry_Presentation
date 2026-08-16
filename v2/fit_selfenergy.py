"""원소별 self-energy(기준 에너지)를 학습 데이터에서 최소제곱으로 적합.

constants.py의 Etot 값을 갱신할 때 쓴다. 학습에 쓰는 데이터셋 조합이 바뀌면
(예: ani1x_ccsdt 제외) 기준도 그 조합으로 다시 적합해야 정합이 맞는다.

E_total(분자) ≈ Σ_Z n_Z · c_Z 를 train split 표본(데이터셋 동등 가중)으로 풀고,
계수 c_Z와 데이터셋별 잔차 σ(= 신경망이 배울 타깃의 크기)를 출력한다.
train split만 쓰므로 val/test 누수 없음. 전처리(npy)가 먼저 되어 있어야 한다.

    python v2/fit_selfenergy.py                                # 기본: ccsdt 제외 4종
    python v2/fit_selfenergy.py qm9x ani2x                     # 지정 조합
"""

import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from train_wandb import DS_ALL, open_split

import constants as con

H2K = 627.5094740631
ZS = [1, 6, 7, 8, 9, 16, 17]
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}
PER_DS = 50_000  # 데이터셋별 표본 수 (동등 가중)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    targets = sys.argv[1:] or ["qm9x", "ani1x_wb97x", "ani2x", "transition1x"]
    bad = [t for t in targets if t not in DS_ALL]
    if bad:
        sys.exit(f"알 수 없는 데이터셋 {bad} (가능: {DS_ALL})")

    rng = np.random.default_rng(0)
    data = {}
    for ds in targets:
        o = open_split(ds, "train")
        if o is None:
            sys.exit(f"{ds}: 전처리 데이터 없음 — python v2/preprocess.py 먼저")
        M = len(o["E"])
        sel = np.sort(rng.choice(M, min(PER_DS, M), replace=False))
        C = np.zeros((len(sel), len(ZS)))
        E = np.zeros(len(sel))
        N = np.zeros(len(sel))
        for i, mi in enumerate(sel):
            a, b = int(o["ptr"][mi]), int(o["ptr"][mi + 1])
            z = np.asarray(o["Z"][a:b])
            for j, zz in enumerate(ZS):
                C[i, j] = (z == zz).sum()
            E[i] = o["E"][mi]
            N[i] = len(z)
        data[ds] = (C, E, N)
        print(f"{ds:14s} 표본 {len(sel):6d}", flush=True)

    C = np.vstack([v[0] for v in data.values()])
    E = np.concatenate([v[1] for v in data.values()])
    coef, *_ = np.linalg.lstsq(C, E, rcond=None)

    print(f"\n=== self-energy 적합 결과 (데이터셋: {', '.join(targets)}) ===")
    print("constants.py Etot에 넣을 값:")
    n_el = C.sum(axis=0)  # 원소별 등장 원자 수
    for j, zz in enumerate(ZS):
        cur = con.Etot[zz]
        if n_el[j] == 0:
            print(f"    # Z={zz} {SYM[zz]}: 표본에 없음 → 적합 불가, 기존 값({cur:.6f}) 유지할 것")
            continue
        print(f"    {coef[j]:.6f},  # Z={zz} {SYM[zz]}   (적합; 현재 constants.py {cur:.6f}, "
              f"차이 {(coef[j] - cur) * H2K:+.2f} kcal/mol)")

    print(f"\n=== 이 계수 기준 데이터셋별 NN 타깃 σ ===")
    for ds, (Cd, Ed, Nd) in data.items():
        r = Ed - Cd @ coef
        print(f"  {ds:14s} σ = {r.std():.4f} Ha ({r.std() * H2K:6.1f} kcal/mol)"
              f"   per-atom {(r / Nd).std():.5f} Ha")


if __name__ == "__main__":
    main()
