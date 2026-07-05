# 추론 I/O 계약 (프론트엔드 · 백엔드 연결용)

**대상:** `model/predict_forces.py`(학습된 NNP 추론 코드)를 `backend/main.py`에 끼워 넣어
프론트(`demo/index.html`)에 서빙하려는 사람.

작성: 모델/학습 담당. 학습된 체크포인트(`checkpoint_best.pt`)와 추론 코드 기준.

---

## TL;DR

- **입력:** 원자 = 원소 + xyz 좌표(Å). **H/C/N/O 4종만 지원.**
- **출력:** 총에너지 + 원자별 힘.
- **단위:** 에너지 = **Hartree**, 힘 = **Hartree/Å**, 좌표 = **Å**.
  (현재 `PredictResponse.unit = "eV"`는 사실과 다름 → 변환하거나 라벨 수정 필요)
- **힘 정의:** `F_i = −∂E/∂r_i` (부호 이미 적용됨).
- **함수:** `predict_forces.predict_energy_forces(Z, R, ckpt=None)`.

---

## 1. 핵심 함수

```python
from predict_forces import load_model, predict_energy_forces
```

### `predict_energy_forces(Z, R, ckpt=None)`
- `Z` : **원자번호** 정수 배열. H=1, C=6, N=7, O=8. shape `(n,)` 또는 배치 `(b,n)`.
  - ⚠️ **원자번호**를 넣는다(원소 기호 아님, 모델 내부 종 인덱스 0~3도 아님). 함수가 내부에서 종 인덱스로 변환.
  - ⚠️ **H/C/N/O 외 원소 금지.** 모델이 이 4종만 학습 — F/P/S/Cl 등 넣으면 안 됨.
- `R` : 좌표(Å). shape `(n,3)` 또는 `(b,n,3)`. float.
- `ckpt` : 생략 시 `model/checkpoints/checkpoint_best.pt`를 자동 로드. **반복 호출이면 `ckpt = load_model()` 결과(튜플)를 넘겨 재로딩 방지**(서버는 기동 시 1회 로드해 재사용).
- **반환:**
  - 단일 입력 → `{"energy_hartree": float, "forces": np.ndarray (n,3)}`
  - 배치 입력 → `{"energy_hartree": np.ndarray (b,), "forces": np.ndarray (b,n,3)}`

### `load_model(ckpt_path=...)` → `(model, ref, std)`
서버 기동 시 1회 호출해 튜플을 들고 있다가 매 요청에 `ckpt=`로 넘긴다.

---

## 2. 단위 / 규약

| 항목 | 단위 / 정의 |
|---|---|
| 좌표 `R` | Å (옹스트롬) |
| `energy_hartree` | Hartree — **총 전자에너지**(원소 self-energy 포함, 큰 음수) |
| `forces` | Hartree/Å |
| 힘 | `F_i = −∂E/∂r_i` (음의 그래디언트) |

**변환:** `1 Hartree = 27.211386 eV = 627.50947 kcal/mol`. 힘도 동일 배수(`Hartree/Å → eV/Å`는 ×27.211386).

**주의:** `energy_hartree`는 **총에너지**(~수백 Ha 음수)다. 화면 표시는 기준 대비 ΔE가 더 의미 있을 수 있음. **동역학/힘에는 절대값 무관**(기울기만 사용).

---

## 3. 체크포인트

- 파일: **`model/checkpoints/checkpoint_best.pt`** (`train.py` 산출물). 기존 위치 `model/checkpoint_best.pt`도 fallback으로 지원.
- 내부: `model_state` + `config(dim/rel_dim/num_atoms)` + `energy_ref`(원소 기준에너지) + `target_std`. `load_model()`이 모두 복원하고, 예측 총에너지 = `out*std + counts@energy_ref`로 역정규화.
- ⚠️ torch 2.6+에서 **`weights_only=False`** 필요(energy_ref가 numpy 배열). `load_model()`에 이미 반영됨.
- **`backend/main.py`의 `MODEL_PATH="model.pt"` / `_PlaceholderNNP`는 쓰지 말고 `predict_forces.load_model()`로 교체.**

---

## 4. `backend/main.py` 연결 가이드 (HTTP 계약은 유지, 내부만 교체)

### 기동 시 (lifespan)
```python
from predict_forces import load_model
_ckpt = load_model()   # (model, ref, std) — 전역으로 보관, 매 요청 재사용
```

### POST /predict — 요청은 그대로
요청: `{ "atoms": [ {"element":"C","x":..,"y":..,"z":..}, ... ] }`
- `element`는 **H/C/N/O만** 허용. 그 외는 400/422로 거절(검증 추가 권장).

`run_inference` 교체:
```python
import numpy as np
from predict_forces import predict_energy_forces
_SYM2Z = {"H": 1, "C": 6, "N": 7, "O": 8}

Z = np.array([_SYM2Z[a.element] for a in req.atoms])           # 원소기호 → 원자번호
R = np.array([[a.x, a.y, a.z] for a in req.atoms], np.float32)  # Å
res = predict_energy_forces(Z, R, ckpt=_ckpt)   # ⚠️ no_grad/half로 감싸지 말 것(힘은 autograd)
energy = float(res["energy_hartree"])           # Hartree
forces = res["forces"].tolist()                 # Hartree/Å, [[fx,fy,fz],...]
# unit을 "Hartree"/"Hartree/Å"로 표기하거나 eV로 변환(×27.211386)
```
응답 필드(energy/forces/n_atoms/unit)는 그대로 두되 **`unit`을 실제 단위로 정확히 표기**.

### POST /step — 힘 기반 한 스텝(최속강하 완화)
`F = −∇E`이므로 **힘 방향으로 움직이면 에너지가 감소** → 완화 데모에 사용. 기존 `_relax_step` 플레이스홀더를 이걸로 교체:
```python
res = predict_energy_forces(Z, R, ckpt=_ckpt)
F = res["forces"]                          # (n,3) Hartree/Å
step = np.clip(req.dt * F, -0.1, 0.1)      # dt 작게 + 노름 클리핑(발산 방지)
R_new = (R + step).tolist()
```
응답: `atoms=R_new`, `energy=float(res["energy_hartree"])`.

---

## 5. 하드 제약 / 함정 (반드시 지킬 것)

- **H/C/N/O만.** 그 외 원소 입력 금지.
- 좌표 단위 **Å**. 함수 인자는 **원자번호**(기호 아님).
- 원자 수 **n ≥ 2 권장**(쌍 상호작용 기반 — n=1은 의미 없음).
- 힘은 **autograd**라 `torch.no_grad()`·`model.half()`(fp16)로 감싸면 안 됨 → **fp32 유지**(현 백엔드의 no_grad/half 경로와 충돌).
- 추론에 **데이터셋(h5) 불필요** — 체크포인트만 있으면 됨.
- 모델은 회전 불변 에너지 · 병진 불변 → **힘 알짜합 ≈ 0**(검증 완료). 정상 동작 sanity check로 쓸 수 있음.

## 6. 모델 정확도(기대치 설정)

- 검증 MAE ≈ **0.057 Hartree (≈ 36 kcal/mol)** — 발표 데모용으로는 충분하나 **화학적 정확도(1~2 kcal/mol)에는 못 미침**.
- 힘은 "모델 에너지의 정확한 미분"임은 검증됐으나(유한차분 일치), **힘의 절대 정확도(실제 DFT 힘 대비)는 미검증** — 완화 방향성 데모엔 OK, 정량적 신뢰는 금물.
