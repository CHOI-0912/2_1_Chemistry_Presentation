# atomic_numbers, ccsd(t)_cbs.energy, coordinates
#
# ANI-1x(ani1x-release.h5)에서 위 3개 feature만 lazy하게 numpy로 읽어온다.
# - 파일 전체(5.6GB)를 메모리에 올리지 않고, generator로 "한 그룹(분자식)씩" yield.
# - 호출부에서 for 루프로 하나씩 받아 쓰면 됨.

import os
import sys
import h5py
import numpy as np

# Windows 콘솔(cp949)에서 한글이 깨지지 않도록 UTF-8 출력
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "ani1x-release.h5")  # 저장소 루트 data/
LEGACY_DATA_PATH = os.path.join(BASE_DIR, "data", "ani1x-release.h5")
PATH = DATA_PATH if os.path.exists(DATA_PATH) else LEGACY_DATA_PATH

# 필요한 feature만
KEYS = ["atomic_numbers", "coordinates", "ccsd(t)_cbs.energy"]

# 원자번호 -> 종(species) 인덱스. 원자번호 오름차순으로 0부터 매칭:
#   H(1)->0, C(6)->1, N(7)->2, O(8)->3   (데이터셋엔 이 4원소만 존재)
_Z2IDX = np.full(9, -1, dtype=np.int8)  # 인덱스 0~8, 미등록 원소는 -1
_Z2IDX[[1, 6, 7, 8]] = [0, 1, 2, 3]


def to_species_index(atomic_numbers):
    """원자번호 배열을 0~3 종 인덱스로 변환 (H=0, C=1, N=2, O=3)."""
    return _Z2IDX[atomic_numbers]


def iter_molecules(path=PATH):
    """그룹(분자식)을 하나씩 lazy하게 순회하며, 필요한 feature만 numpy로 읽어 yield.
    한 번에 한 그룹만 메모리에 올라간다.

    yield 형태(dict):
        formula        : str          그룹 이름(화학식)
        atomic_numbers : (Na,)        종 인덱스 0~3 (H=0, C=1, N=2, O=3) — 원자번호를 덮어씀
        coordinates    : (Nc, Na, 3)  좌표 (Å)
        energy         : (Nc,)        ccsd(t)_cbs.energy (Hartree, 미계산은 NaN)
    """
    with h5py.File(path, "r") as f:
        for name in f.keys():
            grp = f[name]
            # 3개 feature가 모두 있는 그룹만 사용
            if not all(k in grp for k in KEYS):
                continue
            # [:] 시점에 해당 그룹 데이터만 디스크 -> numpy 로 읽힘
            yield {
                "formula": name,
                # 원자번호(1,6,7,8)를 0~3 종 인덱스로 덮어씀
                "atomic_numbers": to_species_index(grp["atomic_numbers"][:]),  # (Na,)
                "coordinates": grp["coordinates"][:],         # (Nc, Na, 3)
                "energy": grp["ccsd(t)_cbs.energy"][:],       # (Nc,)
            }


if __name__ == "__main__":
    # 데이터셋(3개 feature를 모두 가진 그룹들)에 등장하는 모든 원소 종류 수집.
    # generator는 한 번만 만들고 for로 끝까지 순회해야 전체 그룹을 본다.
    # Z2SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
    # elements = set()
    # for mol in iter_molecules():
    #     elements |= set(mol["atomic_numbers"].tolist())
    # elements = sorted(elements)
    # print("등장하는 원자번호:", elements)
    # print("등장하는 원소기호:", [Z2SYM.get(z, z) for z in elements])

    # mol = next(iter_molecules())
    # print(mol['coordinates'])

    # === 전체 데이터셋의 원자쌍(atom-atom) 거리 분포 통계 (pooled) ===
    # 거리 개수가 ~6.3억개라 전부 담지 못함:
    #  - mean/std/min/max : 스트리밍으로 정확히 누적
    #  - 분위수(Q..)      : 균일 표본추출(subsample) 후 근사
    rng = np.random.default_rng(0)
    TARGET = 5_000  # 분위수용 표본 목표 개수

    # 1) 메타데이터(shape)만 읽어 전체 거리 개수 -> 표본 확률 p
    total_pairs = 0
    with h5py.File(PATH, "r") as f:
        for name in f.keys():
            g = f[name]
            if not all(k in g for k in KEYS):
                continue
            Nc, Na, _ = g["coordinates"].shape
            total_pairs += Nc * Na * (Na - 1) // 2
    p = min(1.0, TARGET / total_pairs)

    # 2) 좌표를 lazy하게 읽으며 거리 계산 (스트리밍 통계 + 표본 수집)
    n = 0
    s1 = 0.0  # Σd
    s2 = 0.0  # Σd^2
    dmin, dmax = np.inf, -np.inf
    sample = []
    for mol in iter_molecules():
        xyz = mol["coordinates"].astype(np.float64)  # (Nc, Na, 3)
        Nc, Na, _ = xyz.shape
        iu = np.triu_indices(Na, k=1)  # 상삼각(중복 없는 원자쌍) 인덱스
        chunk = max(1, 2_000_000 // (Na * Na))  # (Nc,Na,Na) 메모리 제한
        for s in range(0, Nc, chunk):
            c = xyz[s:s + chunk]                       # (b, Na, 3)
            diff = c[:, :, None, :] - c[:, None, :, :]  # (b, Na, Na, 3)
            d = np.sqrt((diff * diff).sum(-1))          # (b, Na, Na)
            d = d[:, iu[0], iu[1]].ravel()              # 원자쌍 거리 1D
            n += d.size
            s1 += d.sum()
            s2 += (d * d).sum()
            dmin = min(dmin, d.min())
            dmax = max(dmax, d.max())
            m = rng.random(d.size) < p                  # 분위수용 표본
            if m.any():
                sample.append(d[m])

    mean = s1 / n
    std = np.sqrt(max(0.0, s2 / n - mean * mean))
    sample = np.concatenate(sample)
    q = np.percentile(sample, [5, 25, 50, 75, 95])

    print(f"원자쌍 거리 개수 : {n:,}")
    print(f"평균거리 (Å)     : {mean:.4f}")
    print(f"표준편차 (Å)     : {std:.4f}")
    print(f"최소 (Å)         : {dmin:.4f}")
    print(f"최대 (Å)         : {dmax:.4f}")
    print(f"분위수 표본수    : {sample.size:,}  (p={p:.4g})")
    print(f"Q05 (Å)          : {q[0]:.4f}")
    print(f"Q25 (Å)          : {q[1]:.4f}")
    print(f"Q50 (Å)          : {q[2]:.4f}")
    print(f"Q75 (Å)          : {q[3]:.4f}")
    print(f"Q95 (Å)          : {q[4]:.4f}")
