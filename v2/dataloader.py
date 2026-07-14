"""NNP 학습용 통합 데이터로더.

data/ 아래 4개 데이터셋(ANI-1x, ANI-2x, QM9x, Transition1x)을 하나의 형태로 읽는다.
각 로더는 제너레이터이며 next() 시 conformer 하나를 다음 튜플로 반환한다.

    (name, coordinates (N,3) float32, atomic_numbers (N,) int8,
     energy float, forces (N,3) float32 | None)

단위는 Hartree / Hartree·Å^-1 로 통일한다 (ANI 계열은 원본이 이미 Hartree,
QM9x·Transition1x는 eV라 변환). 자세한 배경은 v2/tradeoffs.md 참고.
"""

from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ANI1X_H5 = DATA_DIR / "ani1x-release.h5"
ANI2X_H5 = DATA_DIR / "ANI-2x-wB97X-631Gd.h5"
QM9X_H5 = DATA_DIR / "qm9x.h5"
TRANSITION1X_H5 = DATA_DIR / "Transition1x" / "transition1x.h5"

EV2HARTREE = 1.0 / 27.211386245988

# QM9x / Transition1x 공식 로더의 원자별 참조 에너지 (eV).
REFERENCE_ENERGIES_EV = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
    9: -2712.8213146878606,
}

_SYMBOLS = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}


def _hill_formula(atomic_numbers):
    """원자번호 배열 → Hill 표기 화학식 문자열 (C, H, 나머지 알파벳순)."""
    counts = {}
    for z in np.asarray(atomic_numbers).ravel():
        sym = _SYMBOLS.get(int(z), f"Z{int(z)}")
        counts[sym] = counts.get(sym, 0) + 1

    order = []
    for sym in ("C", "H"):
        if sym in counts:
            order.append(sym)
    order += sorted(s for s in counts if s not in ("C", "H"))

    return "".join(s if counts[s] == 1 else f"{s}{counts[s]}" for s in order)


def _check_file(path, hint):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. {hint}")
    if path.stat().st_size == 0:
        raise FileNotFoundError(f"{path} 가 0바이트입니다(다운로드 미완료). {hint}")
    return path


def load_ani1x_wb97x(h5path=ANI1X_H5):
    """ANI-1x, ωB97x/6-31G(d) 계열 (wb97x_dz). 에너지 + 힘, Hartree.

    공식 로더(data/ANI1x_datasets/dataloader.py)와 같이 NaN 항목을 걸러낸다.
    """
    _check_file(h5path, "ANI-1x h5 파일을 확인하세요.")
    with h5py.File(h5path, "r") as f:
        for name, grp in f.items():
            energies = grp["wb97x_dz.energy"][()]
            forces = grp["wb97x_dz.forces"][()]
            mask = ~np.isnan(energies)
            mask &= ~np.isnan(forces.reshape(len(energies), -1)).any(axis=1)
            if not mask.any():
                continue

            numbers = grp["atomic_numbers"][()]
            coords = grp["coordinates"][()][mask]
            energies = energies[mask]
            forces = forces[mask]

            for i in range(len(energies)):
                yield name, coords[i], numbers, float(energies[i]), forces[i]


def load_ani1x_ccsdt(h5path=ANI1X_H5):
    """ANI-1x, CCSD(T)/CBS 계열 (ANI-1ccx 서브셋). 힘이 없어 forces=None. Hartree."""
    _check_file(h5path, "ANI-1x h5 파일을 확인하세요.")
    with h5py.File(h5path, "r") as f:
        for name, grp in f.items():
            if "ccsd(t)_cbs.energy" not in grp:
                continue

            energies = grp["ccsd(t)_cbs.energy"][()]
            mask = ~np.isnan(energies)
            if not mask.any():
                continue

            numbers = grp["atomic_numbers"][()]
            coords = grp["coordinates"][()][mask]
            energies = energies[mask]

            for i in range(len(energies)):
                yield name, coords[i], numbers, float(energies[i]), None


def load_ani2x(h5path=ANI2X_H5, chunk=8192):
    """ANI-2x (ωB97X/6-31G(d)). 에너지 + 힘, Hartree.

    그룹 키가 원자 수라 분자 이름이 없으므로 species에서 Hill 화학식을 만든다.
    한 그룹이 100만 conformer를 넘어 통째로 읽으면 메모리를 크게 쓰므로
    chunk 개씩 잘라 읽는다.
    """
    _check_file(h5path, "ANI-2x h5 파일을 확인하세요.")
    with h5py.File(h5path, "r") as f:
        for grp in f.values():
            n_conf = grp["energies"].shape[0]
            for start in range(0, n_conf, chunk):
                stop = min(start + chunk, n_conf)
                coords = grp["coordinates"][start:stop]
                energies = grp["energies"][start:stop]
                forces = grp["forces"][start:stop]
                species = grp["species"][start:stop]

                for i in range(stop - start):
                    numbers = species[i]
                    yield (
                        _hill_formula(numbers),
                        coords[i],
                        numbers,
                        float(energies[i]),
                        forces[i],
                    )


def load_qm9x(h5path=QM9X_H5):
    """QM9x (ωB97x/6-31G(d)). 원자화 에너지(atomization energy) + 힘, Hartree.

    공식 로더와 동일하게 total energy에서 원자별 참조 에너지 합을 뺀다.
    """
    _check_file(h5path, "QM9x h5 파일을 확인하세요.")
    with h5py.File(h5path, "r") as f:
        for name, grp in f.items():
            numbers = grp["atomic_numbers"][()]
            positions = grp["positions"][()]
            energies = grp["energy"][()]
            forces = grp["forces"][()]

            ref = sum(REFERENCE_ENERGIES_EV[int(z)] for z in numbers)

            for i in range(len(energies)):
                atomization_ev = float(energies[i]) - ref
                yield (
                    name,
                    positions[i],
                    numbers,
                    atomization_ev * EV2HARTREE,
                    forces[i] * EV2HARTREE,
                )


def load_transition1x(h5path=TRANSITION1X_H5, split="data"):
    """Transition1x (ωB97x/6-31G(d)). 총 에너지 + 힘, Hartree.

    h5 구조: f[split][formula][rxn] 이 반응 경로(NEB) 위의 구조들을 담고,
    그 아래 reactant / product / transition_state 서브그룹이 끝점을 담는다.
    각 그룹의 키는 positions, atomic_numbers, wB97x_6-31G(d).energy/.forces (eV).
    이름은 "{formula}/{rxn}" 형태로 준다.

    주의: 현재 저장소에는 실제 데이터가 없다 (transition1x.h5 = 0바이트).
    data/Transition1x/download_t1x.py 로 받아야 동작한다. v2/tradeoffs.md 참고.
    """
    if split not in ("data", "train", "val", "test"):
        raise ValueError("split은 'data', 'train', 'val', 'test' 중 하나여야 합니다.")

    _check_file(
        h5path,
        "Transition1x 데이터가 없습니다. "
        "`python data/Transition1x/download_t1x.py data/Transition1x` 로 내려받으세요.",
    )

    with h5py.File(h5path, "r") as f:
        for formula, grp in f[split].items():
            for rxn, subgrp in grp.items():
                name = f"{formula}/{rxn}"
                groups = [subgrp["reactant"], subgrp["product"], subgrp]
                for g in groups:
                    numbers = g["atomic_numbers"][()]
                    positions = g["positions"][()]
                    energies = g["wB97x_6-31G(d).energy"][()]
                    forces = g["wB97x_6-31G(d).forces"][()]

                    for i in range(len(energies)):
                        yield (
                            name,
                            positions[i],
                            numbers,
                            float(energies[i]) * EV2HARTREE,
                            forces[i] * EV2HARTREE,
                        )


LOADERS = {
    "ani1x_wb97x": load_ani1x_wb97x,
    "ani1x_ccsdt": load_ani1x_ccsdt,
    "ani2x": load_ani2x,
    "qm9x": load_qm9x,
    "transition1x": load_transition1x,
}


def load(dataset, **kwargs):
    """이름으로 로더를 얻는다. dataset ∈ LOADERS."""
    if dataset not in LOADERS:
        raise ValueError(f"알 수 없는 데이터셋 '{dataset}'. 가능한 값: {list(LOADERS)}")
    return LOADERS[dataset](**kwargs)


if __name__ == "__main__":
    for dataset in LOADERS:
        try:
            name, coords, numbers, energy, forces = next(load(dataset))
        except (FileNotFoundError, OSError) as err:
            print(f"{dataset:14s} SKIP  {err}")
            continue

        shape = "None" if forces is None else str(tuple(forces.shape))
        print(
            f"{dataset:14s} {name:12s} coords={tuple(coords.shape)} "
            f"Z={tuple(numbers.shape)} E={energy:.6f} Ha F={shape}"
        )
