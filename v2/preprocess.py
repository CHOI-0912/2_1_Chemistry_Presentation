"""5개 소스 h5 → data/processed/{dataset}/{split}/ ragged npy (mmap 학습용).

분자마다 원자 수가 달라 패딩 대신 연접(ragged) 저장: 분자 i = ptr[i]:ptr[i+1] 구간.

  energy.npy   f64 (M,)    Hartree, total energy
  ptr.npy      i64 (M+1,)  ptr[0]=0
  numbers.npy  u8  (ΣN,)   원자번호
  coords.npy   f32|f64 (ΣN,3)  Å      — 소스 dtype 그대로 (무손실)
  forces.npy   f32|f64 (ΣN,3)  Ha/Å   — ani1x_ccsdt는 파일 없음
  formula_id.npy i32 (M,) + formulas.json   — 조성별 평가·감사용
  [t1x만] kind.npy u8 (0=경로, 1=reactant, 2=product, 3=TS), rxn_id.npy i32 + rxns.json
  meta.json    개수·단위·생성일

분할: qm9x/ani 계열은 splits.json(조성 90:5:5), transition1x는 h5 내장 공식 분할.
eV 소스(qm9x·t1x)는 f64 곱셈 1회로 Hartree 변환(상대오차 ~1e-16).
t1x는 경로 행만 저장 — reactant/product/TS가 경로에 1회씩 이미 포함돼 있어(실측) 중복 저장 안 함.

사용:  python v2/preprocess.py            # 전부
       python v2/preprocess.py qm9x ani2x # 일부만
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import h5py
import numpy as np
from numpy.lib.format import open_memmap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import ANI1X_H5, ANI2X_H5, QM9X_H5, TRANSITION1X_H5, EV2HARTREE, _hill_formula
from splits import SPLITS, SPLITS_JSON, canon

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
ASSIGN = json.loads(SPLITS_JSON.read_text(encoding="utf-8"))["assign"]
T1X_K = "wB97x_6-31G(d)"


class Writer:
    def __init__(self, ds, split, M, SN, r_dt, f_dt, t1x=False):
        d = PROC / ds / split
        d.mkdir(parents=True, exist_ok=True)
        self.d, self.ds, self.split = d, ds, split
        self.E = open_memmap(d / "energy.npy", mode="w+", dtype=np.float64, shape=(M,))
        self.ptr = open_memmap(d / "ptr.npy", mode="w+", dtype=np.int64, shape=(M + 1,))
        self.ptr[0] = 0
        self.Z = open_memmap(d / "numbers.npy", mode="w+", dtype=np.uint8, shape=(SN,))
        self.R = open_memmap(d / "coords.npy", mode="w+", dtype=r_dt, shape=(SN, 3))
        self.F = open_memmap(d / "forces.npy", mode="w+", dtype=f_dt, shape=(SN, 3)) if f_dt else None
        self.fid = open_memmap(d / "formula_id.npy", mode="w+", dtype=np.int32, shape=(M,))
        if t1x:
            self.kind = open_memmap(d / "kind.npy", mode="w+", dtype=np.uint8, shape=(M,))
            self.rxn_id = open_memmap(d / "rxn_id.npy", mode="w+", dtype=np.int32, shape=(M,))
            self.rmap = {}
        else:
            self.kind = None
        self.fmap = {}
        self.m = self.n = 0

    def add_block(self, formula, numbers, coords, energies, forces, kind=None, rxn=None):
        """같은 원자 배열을 공유하는 conformer 블록. coords (k,N,3)."""
        k, N = coords.shape[0], coords.shape[1]
        self.E[self.m:self.m + k] = energies
        self.fid[self.m:self.m + k] = self.fmap.setdefault(formula, len(self.fmap))
        self.ptr[self.m + 1:self.m + k + 1] = self.n + np.arange(1, k + 1, dtype=np.int64) * N
        self.Z[self.n:self.n + k * N] = np.tile(np.asarray(numbers, np.uint8), k)
        self.R[self.n:self.n + k * N] = coords.reshape(-1, 3)
        if self.F is not None:
            self.F[self.n:self.n + k * N] = forces.reshape(-1, 3)
        if self.kind is not None:
            self.kind[self.m:self.m + k] = kind
            self.rxn_id[self.m:self.m + k] = self.rmap.setdefault(rxn, len(self.rmap))
        self.m += k
        self.n += k * N

    def close(self, source):
        assert self.m == len(self.E) and self.n == len(self.Z), \
            f"{self.ds}/{self.split} 개수 불일치: {self.m}/{len(self.E)}, {self.n}/{len(self.Z)}"
        (self.d / "formulas.json").write_text(
            json.dumps(sorted(self.fmap, key=self.fmap.get)), encoding="utf-8")
        if self.kind is not None:
            (self.d / "rxns.json").write_text(
                json.dumps(sorted(self.rmap, key=self.rmap.get)), encoding="utf-8")
        (self.d / "meta.json").write_text(json.dumps({
            "source": source, "created": date.today().isoformat(),
            "units": {"energy": "Hartree", "coords": "Angstrom", "forces": "Hartree/Angstrom"},
            "energy_kind": "total", "n_mol": self.m, "n_atoms_total": self.n,
            "has_forces": self.F is not None,
        }), encoding="utf-8")
        print(f"  {self.ds}/{self.split}: 분자 {self.m:,} / 원자 {self.n:,}", flush=True)


def build_qm9x():
    with h5py.File(QM9X_H5, "r") as f:
        cnt = {s: [0, 0] for s in SPLITS}
        for name, grp in f.items():
            s = ASSIGN[canon(name)]
            k, N = grp["energy"].shape[0], grp["atomic_numbers"].shape[0]
            cnt[s][0] += k; cnt[s][1] += k * N
        ws = {s: Writer("qm9x", s, *cnt[s], np.float64, np.float64) for s in SPLITS}
        for name, grp in f.items():
            s = ASSIGN[canon(name)]
            ws[s].add_block(canon(name), grp["atomic_numbers"][()],
                            grp["positions"][()],
                            grp["energy"][()].astype(np.float64) * EV2HARTREE,
                            grp["forces"][()].astype(np.float64) * EV2HARTREE)
    for w in ws.values():
        w.close("qm9x.h5 (eV→Ha 변환)")


def build_ani1x(tag):
    """tag ∈ {ani1x_wb97x, ani1x_ccsdt}. NaN 항목은 공식 로더처럼 거른다."""
    ekey = "wb97x_dz.energy" if tag == "ani1x_wb97x" else "ccsd(t)_cbs.energy"
    fkey = "wb97x_dz.forces" if tag == "ani1x_wb97x" else None

    def valid_mask(grp):
        if ekey not in grp:
            return None
        m = ~np.isnan(grp[ekey][()])
        if fkey:
            m &= ~np.isnan(grp[fkey][()].reshape(len(m), -1)).any(axis=1)
        return m if m.any() else None

    with h5py.File(ANI1X_H5, "r") as f:
        cnt = {s: [0, 0] for s in SPLITS}
        for name, grp in f.items():
            m = valid_mask(grp)
            if m is None:
                continue
            s = ASSIGN[canon(name)]
            k, N = int(m.sum()), grp["atomic_numbers"].shape[0]
            cnt[s][0] += k; cnt[s][1] += k * N
        ws = {s: Writer(tag, s, *cnt[s], np.float32, np.float32 if fkey else None) for s in SPLITS}
        for name, grp in f.items():
            m = valid_mask(grp)
            if m is None:
                continue
            s = ASSIGN[canon(name)]
            ws[s].add_block(canon(name), grp["atomic_numbers"][()],
                            grp["coordinates"][()][m], grp[ekey][()][m],
                            grp[fkey][()][m] if fkey else None)
    for w in ws.values():
        w.close(f"ani1x-release.h5 [{ekey}]")


def build_ani2x(chunk=65536):
    hill = {}  # species row bytes → (formula, split)

    def key_of(row):
        b = row.tobytes()
        hit = hill.get(b)
        if hit is None:
            fm = _hill_formula(row)
            hit = (fm, ASSIGN[fm])
            hill[b] = hit
        return hit

    with h5py.File(ANI2X_H5, "r") as f:
        cnt = {s: [0, 0] for s in SPLITS}
        for grp in f.values():
            n_conf, N = grp["species"].shape
            for st in range(0, n_conf, chunk):
                sp = grp["species"][st:st + chunk]
                uniq, counts = np.unique(sp, axis=0, return_counts=True)
                for row, k in zip(uniq, counts):
                    s = key_of(row)[1]
                    cnt[s][0] += int(k); cnt[s][1] += int(k) * N
        ws = {s: Writer("ani2x", s, *cnt[s], np.float32, np.float64) for s in SPLITS}
        for grp in f.values():
            n_conf, N = grp["species"].shape
            for st in range(0, n_conf, chunk):
                sp = grp["species"][st:st + chunk]
                co = grp["coordinates"][st:st + chunk]
                en = grp["energies"][st:st + chunk]
                fo = grp["forces"][st:st + chunk]
                uniq, inv = np.unique(sp, axis=0, return_inverse=True)
                for ui, row in enumerate(uniq):
                    fm, s = key_of(row)
                    sel = inv == ui
                    ws[s].add_block(fm, row, co[sel], en[sel], fo[sel])
    for w in ws.values():
        w.close("ANI-2x-wB97X-631Gd.h5")


def build_transition1x():
    with h5py.File(TRANSITION1X_H5, "r") as f:
        for s in SPLITS:
            cnt = [0, 0]
            for formula, grp in f[s].items():
                for rxn, sub in grp.items():
                    k, N = sub[T1X_K + ".energy"].shape[0], sub["atomic_numbers"].shape[0]
                    cnt[0] += k; cnt[1] += k * N
            w = Writer("transition1x", s, *cnt, np.float64, np.float64, t1x=True)
            unmatched = 0
            for formula, grp in f[s].items():
                for rxn, sub in grp.items():
                    e = sub[T1X_K + ".energy"][()]
                    kind = np.zeros(len(e), dtype=np.uint8)
                    # reactant/product/TS는 경로에 정확히 1회씩 포함(실측) → 에너지 일치로 마킹
                    for kv, key in ((1, "reactant"), (2, "product"), (3, "transition_state")):
                        hit = np.where(e == float(sub[key][T1X_K + ".energy"][0]))[0]
                        if len(hit) == 1:
                            kind[hit[0]] = kv
                        else:
                            unmatched += 1
                    w.add_block(canon(formula), sub["atomic_numbers"][()],
                                sub["positions"][()],
                                e.astype(np.float64) * EV2HARTREE,
                                sub[T1X_K + ".forces"][()].astype(np.float64) * EV2HARTREE,
                                kind=kind, rxn=f"{formula}/{rxn}")
            if unmatched:
                print(f"  [주의] t1x/{s}: 끝점-경로 에너지 매칭 실패 {unmatched}건 (kind=0으로 남음)")
            w.close("Transition1x.h5 (공식 분할, eV→Ha 변환)")


BUILDERS = {
    "qm9x": build_qm9x,
    "ani1x_wb97x": lambda: build_ani1x("ani1x_wb97x"),
    "ani1x_ccsdt": lambda: build_ani1x("ani1x_ccsdt"),
    "ani2x": build_ani2x,
    "transition1x": build_transition1x,
}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    targets = sys.argv[1:] or list(BUILDERS)
    for t in targets:
        if t not in BUILDERS:
            sys.exit(f"알 수 없는 데이터셋 {t!r}. 가능: {list(BUILDERS)}")
        t0 = time.time()
        print(f"[{t}] 시작", flush=True)
        BUILDERS[t]()
        print(f"[{t}] 완료 ({time.time() - t0:.0f}s)", flush=True)
