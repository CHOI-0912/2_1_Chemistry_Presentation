# ANI-1x 데이터셋(ani1x-release.h5) 탐색 스크립트
#
# 5.6GB짜리 HDF5 파일을 "조금만" 들여다본다.
# h5py는 기본적으로 lazy(슬라이스할 때만 디스크에서 읽음)이므로,
# 키 목록 / shape / 작은 슬라이스만 읽어 파일 전체를 메모리에 올리지 않는다.
#
# 실행: python explore_ani1x.py

import os
import sys
import h5py
import numpy as np

# Windows 콘솔(cp949)에서 'Å'·한글이 깨지거나 크래시하지 않도록 UTF-8로 출력
sys.stdout.reconfigure(encoding="utf-8")

# 정리된 data/ 폴더를 우선 사용하고, 기존 위치도 호환한다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "ani1x-release.h5")  # 저장소 루트 data/
LEGACY_DATA_PATH = os.path.join(BASE_DIR, "data", "ani1x-release.h5")
PATH = DATA_PATH if os.path.exists(DATA_PATH) else LEGACY_DATA_PATH

# ANI-1x는 H, C, N, O 4원소로만 구성됨
Z2SYM = {1: "H", 6: "C", 7: "N", 8: "O"}


def main():
    with h5py.File(PATH, "r") as f:
        # 1) 최상위 구조: 분자식별 그룹들 (예: 'C1H4', 'C2H6O1', ...)
        groups = list(f.keys())
        print(f"파일: {os.path.basename(PATH)}")
        print(f"최상위 그룹(분자식) 수: {len(groups)}")
        print(f"예시 그룹 이름: {groups[:8]}")

        # 2) 대표 그룹 1개를 골라 실제 스키마(키·shape·dtype) 출력
        name = groups[0]
        grp = f[name]
        print(f"\n=== 대표 그룹 '{name}' 스키마 ===")
        for k in grp.keys():
            d = grp[k]
            print(f"  {k:34s} shape={str(d.shape):20s} dtype={d.dtype}")

        # 3) 샘플 값: 원자 구성 + 0번 conformer의 좌표/에너지/힘
        z = grp["atomic_numbers"][:]
        atoms = [Z2SYM.get(int(x), str(x)) for x in z]
        n_atoms = len(z)
        n_conf = grp["coordinates"].shape[0]
        print(f"\n=== 대표 그룹 '{name}' 샘플 ===")
        print(f"원자 수 Na={n_atoms}, conformer 수 Nc={n_conf}")
        print(f"원자 구성: {atoms}")

        print("\n0번 conformer 좌표 (Å):")
        print(grp["coordinates"][0])

        if "wb97x_dz.energy" in grp:
            print(f"\n0번 conformer 에너지 wb97x_dz (Hartree): {grp['wb97x_dz.energy'][0]}")
        if "wb97x_dz.forces" in grp:
            print("0번 conformer 힘 wb97x_dz (Hartree/Å), 앞 3원자:")
            print(grp["wb97x_dz.forces"][0][:3])

        # 4) 고정밀 물성은 일부 conformer만 계산됨 → 나머지는 NaN
        for key in ["ccsd(t)_cbs.energy", "wb97x_tz.energy"]:
            if key in grp:
                e = grp[key][:]
                n_valid = int(np.count_nonzero(~np.isnan(e)))
                print(f"\n'{key}': 계산된(non-NaN) conformer {n_valid} / 전체 {len(e)}")


if __name__ == "__main__":
    main()
