"""조성(formula) 단위 train/val/test 분할 — 90 : 5 : 5 (허용 오차 ±0.5%p).

conformer를 무작위로 나누면 같은 분자의 다른 자세가 train/test에 갈라져 누수가 생긴다
(v2/tradeoffs.md '평가할 때의 함정' 참고). 분할 단위는 조성(화학식)이다 — 같은 조성
(이성질체·conformer 포함)은 통째로 같은 split에 들어간다.

조성마다 conformer 수가 1~수천 개로 들쭉날쭉해 단순 해시 배정으로는 conformer 비율이
목표에서 크게 벗어난다. 그래서 `python v2/splits.py`(빌드)가

  1. 데이터셋별로 조성 → conformer 수를 실측하고
  2. 표기를 정규화한다 (QM9x 'C2F3H3N2'(알파벳순), ANI-1x 'C10H10N2O1'(1 표기),
     ANI-2x Hill이 제각각 → canonical Hill로 통일. 같은 조성이면 데이터셋이 달라도
     같은 split을 받아, 섞어 학습해도 교차 누수가 없다)
  3. 결정적 greedy(큰 조성부터 목표 잔여량이 큰 split에) + 보정 패스로
     "모든" 데이터셋의 conformer 비율을 90:5:5 ±0.5%p 안에 맞춘 뒤
  4. 배정표를 v2/splits.json 에 쓴다.

Transition1x는 h5 안의 공식 분할(train/val/test ≈ 94:3:3, 반응 단위)을 그대로 쓴다.
공식 분할은 분자식 기준으로도 서로소임을 실측으로 확인했고(교집합 0), 전역 배정표는
T1x의 분자식 171개를 공식 배정과 같은 split에 **고정**한 뒤 나머지를 최적화한다 —
그래서 T1x를 포함한 다섯 소스 전부에서 데이터셋 간 교차 유출이 없다.

사용:
    from splits import load_split
    for name, coords, numbers, energy, forces in load_split("qm9x", "train"):
        ...
"""

import json
import re
from pathlib import Path

from dataloader import load

RATIO = (0.90, 0.05, 0.05)  # train : val : test
TOL = 0.005                 # conformer 비율 허용 오차 (±0.5%p)
SPLITS = ("train", "val", "test")
SPLITS_JSON = Path(__file__).resolve().parent / "splits.json"

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def canon(name):
    """조성 문자열 표기를 canonical Hill(C, H, 나머지 알파벳순, 1 생략)로 정규화."""
    counts = {}
    for sym, num in _FORMULA_RE.findall(name):
        counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
    order = [s for s in ("C", "H") if s in counts]
    order += sorted(s for s in counts if s not in ("C", "H"))
    return "".join(s if counts[s] == 1 else f"{s}{counts[s]}" for s in order)


_ASSIGN = None


def _table():
    global _ASSIGN
    if _ASSIGN is None:
        if not SPLITS_JSON.exists():
            raise FileNotFoundError(
                f"{SPLITS_JSON} 이 없습니다. `python v2/splits.py` 로 먼저 생성하세요."
            )
        _ASSIGN = json.loads(SPLITS_JSON.read_text(encoding="utf-8"))["assign"]
    return _ASSIGN


def load_split(dataset, split, **kwargs):
    """dataloader.load()와 같은 튜플을 내주되 해당 split의 샘플만 통과시킨다.

    필터 방식이라 전체를 스트리밍하며 걸러낸다(val/test도 전체 읽기 비용은 같다).
    transition1x는 자체 공식 분할로 위임한다.
    """
    if split not in SPLITS:
        raise ValueError(f"split은 {SPLITS} 중 하나여야 합니다.")
    if dataset == "transition1x":
        yield from load(dataset, split=split, **kwargs)
        return
    table = _table()
    for sample in load(dataset, **kwargs):
        key = canon(sample[0])
        try:
            s = table[key]
        except KeyError:
            raise KeyError(
                f"조성 {key!r}가 splits.json에 없습니다. 데이터가 바뀌었으면 "
                "`python v2/splits.py` 로 배정표를 다시 만드세요."
            ) from None
        if s == split:
            yield sample


# ----------------------------------------------------------------------
# 빌드: 실측 → greedy 배정 → 보정 → splits.json
# ----------------------------------------------------------------------

def _scan_counts():
    """데이터셋(metric)별 {canonical formula: conformer 수}. 수 분 소요(ANI-2x 스캔)."""
    import h5py
    import numpy as np

    from dataloader import ANI1X_H5, ANI2X_H5, QM9X_H5, _hill_formula

    counts = {m: {} for m in ("qm9x", "ani1x_wb97x", "ani1x_ccsdt", "ani2x")}

    with h5py.File(QM9X_H5, "r") as f:
        for name, grp in f.items():
            key = canon(name)
            counts["qm9x"][key] = counts["qm9x"].get(key, 0) + grp["energy"].shape[0]

    # NaN 에너지만 거른다 (wb97x_dz forces에는 실측상 NaN 없음 — tradeoffs.md)
    with h5py.File(ANI1X_H5, "r") as f:
        for name, grp in f.items():
            key = canon(name)
            for tag, dset in (("ani1x_wb97x", "wb97x_dz.energy"), ("ani1x_ccsdt", "ccsd(t)_cbs.energy")):
                if dset not in grp:
                    continue
                n = int((~np.isnan(grp[dset][()])).sum())
                if n:
                    counts[tag][key] = counts[tag].get(key, 0) + n

    with h5py.File(ANI2X_H5, "r") as f:
        cache = {}  # species row bytes → formula (이미 canonical Hill)
        for grp in f.values():
            n_conf = grp["species"].shape[0]
            for start in range(0, n_conf, 8192):
                sp = grp["species"][start : start + 8192]
                uniq, cnt = np.unique(sp, axis=0, return_counts=True)
                for row, k in zip(uniq, cnt):
                    key = cache.get(row.tobytes())
                    if key is None:
                        key = _hill_formula(row)
                        cache[row.tobytes()] = key
                    counts["ani2x"][key] = counts["ani2x"].get(key, 0) + int(k)

    return counts


def _t1x_formula_splits():
    """Transition1x 공식 분할의 분자식 → split. 공식 분할이 분자식-서로소인지도 검증."""
    import h5py

    from dataloader import TRANSITION1X_H5

    fixed = {}
    with h5py.File(TRANSITION1X_H5, "r") as f:
        for s in SPLITS:
            for k in f[s].keys():
                fm = canon(k)
                if fixed.get(fm, s) != s:
                    raise RuntimeError(
                        f"T1x 공식 분할에서 분자식 {fm!r}가 여러 split에 나타남 — 고정 불가"
                    )
                fixed[fm] = s
    return fixed


def _build_assignment(counts, fixed=None):
    """greedy + 보정으로 조성 → split 배정. 모든 metric의 비율을 RATIO ±TOL에 맞춘다.

    fixed(조성 → split)는 그대로 고정하고(T1x 공식 분할 정합용) 나머지만 최적화한다.
    """
    metrics = list(counts)
    totals = {m: sum(counts[m].values()) for m in metrics}
    formulas = sorted({f for m in metrics for f in counts[m]})
    weight = {f: {m: counts[m].get(f, 0) for m in metrics} for f in formulas}

    # 큰 조성부터: 전체 대비 비중 합이 큰 순 (동률이면 이름순 — 결정적)
    formulas.sort(key=lambda f: (-sum(weight[f][m] / totals[m] for m in metrics), f))

    cur = {m: {s: 0 for s in SPLITS} for m in metrics}
    assign = {}
    if fixed:
        for fm, s in fixed.items():
            assign[fm] = s
            for m in metrics:
                cur[m][s] += counts[m].get(fm, 0)
        formulas = [f for f in formulas if f not in assign]

    def dev_after(f, s):
        """f를 s에 넣었을 때, 최종 목표 conformer 수 대비 제곱 편차 합."""
        d = 0.0
        for m in metrics:
            for t in SPLITS:
                c = cur[m][t] + (weight[f][m] if t == s else 0)
                d += ((c - RATIO[SPLITS.index(t)] * totals[m]) / totals[m]) ** 2
        return d

    for f in formulas:
        best = min(SPLITS, key=lambda s: dev_after(f, s))
        assign[f] = best
        for m in metrics:
            cur[m][best] += weight[f][m]

    # 보정: "허용 오차 초과분"의 제곱합을 목적함수로, 줄어드는 단일 이동이면 즉시 수용.
    # (최대 편차만 좇으면 국소 최적에 잘 걸린다 — 초과분 합은 부드러워서 잘 내려간다.)
    sidx = {s: i for i, s in enumerate(SPLITS)}

    def excess(m, s):
        d = abs(cur[m][s] / totals[m] - RATIO[sidx[s]])
        return max(0.0, d - TOL) ** 2

    def total_excess():
        return sum(excess(m, s) for m in metrics for s in SPLITS)

    for _pass in range(200):
        moved = False
        for f in formulas:
            wf = weight[f]
            src = assign[f]
            for dst in SPLITS:
                if dst == src:
                    continue
                affected = [m for m in metrics if wf[m]]
                before = sum(excess(m, src) + excess(m, dst) for m in affected)
                for m in affected:
                    cur[m][src] -= wf[m]
                    cur[m][dst] += wf[m]
                after = sum(excess(m, src) + excess(m, dst) for m in affected)
                if after < before - 1e-18:
                    assign[f] = dst
                    src = dst
                    moved = True
                else:
                    for m in affected:
                        cur[m][src] += wf[m]
                        cur[m][dst] -= wf[m]
        if not moved or total_excess() == 0.0:
            break

    return assign, cur, totals


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    # 조성별 conformer 수 실측은 비싸서(ANI-2x 스캔 수 분) splits.json에 캐시한다.
    # 데이터가 바뀌었으면 `python v2/splits.py --rescan` 으로 강제 재실측.
    counts = None
    if SPLITS_JSON.exists() and "--rescan" not in sys.argv:
        counts = json.loads(SPLITS_JSON.read_text(encoding="utf-8")).get("counts")
        if counts:
            print("splits.json의 실측 카운트 재사용 (--rescan 으로 갱신 가능)", flush=True)
    if not counts:
        print("데이터셋 스캔 중... (ANI-2x 때문에 수 분 걸림)", flush=True)
        counts = _scan_counts()

    fixed = _t1x_formula_splits()
    print(f"T1x 공식 분할 정합: 분자식 {len(fixed)}개 고정", flush=True)
    assign, cur, totals = _build_assignment(counts, fixed)

    SPLITS_JSON.write_text(
        json.dumps({"ratio": RATIO, "tol": TOL, "counts": counts, "assign": assign}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{SPLITS_JSON} 저장 (조성 {len(assign)}개)\n", flush=True)

    ok_all = True
    for m in counts:
        parts, ok = [], True
        for i, s in enumerate(SPLITS):
            share = cur[m][s] / totals[m]
            if abs(share - RATIO[i]) > TOL:
                ok = ok_all = False
            parts.append(f"{s} {cur[m][s]:>9,} ({share * 100:5.2f}%)")
        print(f"{m:14s} " + "  ".join(parts) + f"  | 합계 {totals[m]:,}  | {'PASS' if ok else 'FAIL'}")
    print(f"\n±{TOL * 100:.1f}%p 기준: {'전부 통과' if ok_all else '실패 있음'}")
    print("transition1x   공식 분할 사용(94.27 / 2.80 / 2.94, 반응 단위 9,561/225/287) — 재분할 안 함")
