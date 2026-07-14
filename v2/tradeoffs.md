# 데이터로더 trade-off 정리

`v2/dataloader.py`가 `data/` 아래 4개 데이터셋을 하나의 반환형으로 통일하면서 내린 결정과,
그 때문에 감수한 손실을 기록한다. 반환형은 conformer 하나당

```
(name, coordinates (N,3), atomic_numbers (N,), energy, forces (N,3) | None)
```

## 데이터셋 실측 요약

| 데이터셋 | conformer 수 | 원소 | 이론 수준 | 에너지 종류 | 힘 | 원본 단위 |
|---|---|---|---|---|---|---|
| ANI-1x (wb97x_dz) | 4,956,005 | H C N O | ωB97x/6-31G(d) | total | O | Hartree |
| ANI-1x (ccsd(t)_cbs) | 489,571 | H C N O | CCSD(T)/CBS | total | **X** | Hartree |
| ANI-2x | 9,651,712 | H C N O **F S Cl** | ωB97X/6-31G(d) | total | O | Hartree |
| QM9x | 133,885 | H C N O F | ωB97x/6-31G(d) | **atomization** | O | eV |
| Transition1x | (데이터 없음) | H C N O | ωB97x/6-31G(d) | total | O | eV |

## Transition1x 분석 — 이 저장소에는 실제 데이터가 없다

`data/Transition1x/`에는 공식 저장소(코드·예제·setup.py)만 클론되어 있고 분자 데이터가 없다.

- `data/Transition1x/transition1x.h5` — **0바이트**. 다운로드가 시작만 되고 끊긴 빈 파일이다.
- `data/Transition1x/data/t1x_splits.tar.gz` (213MB) — 압축을 풀어도 **분자 데이터가 아니다.**
  내용은 `splits/split_0..9/{train,val,test}_{idx,formulas}.json`, 즉 10-fold 교차검증용
  **분할 인덱스 목록**뿐이다. 실제 좌표·에너지·힘은 여기에 들어 있지 않다.
- 실제 h5는 figshare에서 받아야 한다: `python data/Transition1x/download_t1x.py data/Transition1x`
  (`download_t1x.py`가 가리키는 URL: figshare file 36035789).

그래서 `load_transition1x()`는 **공식 로더(`data/Transition1x/transition1x/dataloader.py`)와
공식 README의 스펙에 맞춰 미리 구현만 해 두고**, 파일이 없거나 0바이트면 다운로드 방법을 담은
`FileNotFoundError`를 던진다. 데이터를 받는 즉시 다른 로더와 똑같이 동작한다.

**Trade-off:** 실데이터로 검증하지 못했다. 구조(`f[split][formula][rxn]` → `reactant`/`product`
서브그룹 + 경로 상의 구조들, 키는 `positions`·`atomic_numbers`·`wB97x_6-31G(d).energy`·`.forces`)는
공식 코드에서 읽어낸 것이므로 스키마가 바뀌지 않았다면 맞지만, 다른 로더들처럼 실제 shape·NaN 여부를
눈으로 확인하지는 못했다. 데이터를 받으면 `python v2/dataloader.py` 스모크 테스트로 먼저 확인할 것.

내용상 Transition1x는 다른 셋과 성격이 다르다. ANI/QM9x가 **평형 근처 구조**를 샘플링한 데이터라면,
Transition1x는 NEB로 얻은 **반응 경로 위의 구조**(반응물 → 전이상태 → 생성물)다. 전이상태처럼 결합이
끊어지고 만들어지는 영역은 다른 데이터셋에 거의 없어서, 반응을 다루려면 이 데이터가 사실상 필수다.
반대로 이것만으로 학습하면 평형 구조 정확도가 떨어진다.

## 단위 — Hartree로 통일

ANI 계열은 Hartree, QM9x·Transition1x는 eV라 그대로 섞으면 27배 차이가 난다.
로더에서 **Hartree / Hartree·Å⁻¹로 통일**했다 (`EV2HARTREE = 1/27.211386245988`).

- **얻은 것:** 데이터셋을 섞어도 에너지·힘 스케일이 일관된다.
- **잃은 것:** QM9x·Transition1x는 float 변환이 한 번 더 들어간다(부동소수점 오차는 무시할 수준).
  eV가 필요하면 `energy / EV2HARTREE`로 되돌리면 된다.

## 에너지 기준이 데이터셋마다 다르다 (가장 큰 함정)

**QM9x만 원자화 에너지(atomization energy)**를 반환한다. 공식 로더가 `energy - Σ(원자별 참조에너지)`로
정의하고 사용자도 그것을 쓰기로 해서 그대로 따랐다. 나머지 셋은 **총 에너지(total energy)**다.

즉 QM9x의 -1.5 Ha와 ANI-1x의 -386.9 Ha는 **같은 물리량이 아니다.** 두 데이터를 그냥 섞어 학습하면
모델이 배우는 건 화학이 아니라 "어느 데이터셋에서 왔는가"가 된다.

섞어 쓰려면 둘 중 하나로 맞춰야 한다:

- 다른 셋도 원자화 에너지로 바꾼다 — 단, ANI/T1x의 참조 에너지는 각자의 이론 수준에서 계산한 고립 원자
  에너지여야 한다. `REFERENCE_ENERGIES_EV`(H·C·N·O·F)는 QM9x/Transition1x용 값이라 ANI에 그대로 쓰면 안 된다.
- 또는 데이터셋별 self-energy를 선형회귀로 뽑아 빼는 방식(NNP 학습에서 흔한 처리)을 쓴다.

여기서는 로더 단계에서 임의로 통일하지 않고 **원 데이터셋의 정의를 보존**했다. 어떤 기준으로 맞출지는
학습 코드가 결정할 문제이고, 잘못된 참조 에너지를 로더가 몰래 끼워 넣는 것보다 낫다고 봤다.

## ANI-1x: wb97x_dz 채택, ccsd(t)는 별도 로더

- `load_ani1x_wb97x()` — `wb97x_dz.energy/.forces`. tz(def2-TZVPP)도 있지만 dz를 골랐다.
  ANI-2x·QM9x·Transition1x가 전부 ωB97x/6-31G(d)라 **이론 수준이 같아 섞기 좋기 때문**이다.
  대신 tz의 더 높은 정확도는 포기했다.
- `load_ani1x_ccsdt()` — `ccsd(t)_cbs.energy`. **힘이 없어 forces=None**을 반환한다.
  힘 손실(force loss)을 쓰는 학습 루프는 이 로더에서 None을 반드시 처리해야 한다.
- 유효 샘플 수 차이가 크다: wb97x_dz는 495만 전부 유효(NaN 없음), **ccsd(t)_cbs는 48.9만 (약 10%)**,
  3,114개 그룹 중 1,910개에만 값이 있다. CCSD(T)는 비싸서 일부만 계산했기 때문이다.
  → 고정확도 데이터로 파인튜닝하는 용도로는 쓸 수 있지만, 단독 학습용으로는 양이 부족하다.
- 공식 로더처럼 NaN 마스킹을 유지했다. 실측상 wb97x_dz에는 NaN이 없었지만, 데이터셋 문서가 NaN 존재를
  전제하므로 방어적으로 남겨 뒀다(비용은 그룹당 mask 계산 한 번).

## ANI-2x: 분자 이름이 없다

ANI-2x의 h5 그룹 키는 화학식이 아니라 **원자 수**('002' ~ '063')이고, `species`가 conformer마다
행으로 들어 있다. 이름 필드를 채우려고 `species`에서 **Hill 표기 화학식**을 만들었다
(예: `[6,6,1,1,7,7,8,8,8,8]` → `C2H2N2O4`).

**Trade-off:** 이 이름은 조성(composition)일 뿐 **분자를 식별하지 못한다.** 이성질체는 같은 이름이 되고,
원본 데이터셋에 있던 식별자를 복원한 것도 아니다. 이름으로 그룹핑하거나 중복 제거를 하면 안 된다.
conformer마다 문자열을 만드는 비용도 든다(대량 스트리밍 시 병목이 되면 캐싱하거나 이름을 원자수로 대체할 것).

또 ANI-2x는 **F·S·Cl을 포함**한 유일한 데이터셋이다. 다른 셋과 섞어 학습하면 이 세 원소는 ANI-2x에서만
배우게 되므로, 원소별 데이터 불균형을 감안해야 한다.

## ANI-2x 청크 읽기

'013' 그룹 하나에만 conformer가 102만 개다. 공식 예제처럼 그룹을 통째로 읽으면 좌표·힘만으로도
수백 MB가 한 번에 올라온다. `chunk=8192`씩 h5를 슬라이스해 읽는다.

- **얻은 것:** 메모리 사용이 상수로 유지되어 965만 conformer를 스트리밍할 수 있다.
- **잃은 것:** h5 접근 횟수가 늘어 조금 느리다. 메모리가 넉넉하면 `chunk`를 키우면 된다.

## 로더가 하지 않는 것

의도적으로 뺐다. 필요하면 학습 코드에서 할 일이다.

- **셔플·배치·패딩 없음.** 전부 순차 제너레이터다. 원자 수가 제각각이라 배치를 만들려면 패딩이나
  그래프 배칭이 필요한데, 모델 구조에 종속적이라 로더에 넣지 않았다.
- **train/val/test 분할 없음.** Transition1x만 공식 분할(`t1x_splits.tar.gz`)이 있고 나머지는 없다.
  Transition1x 로더는 `split=` 인자로 h5 내부 분할을 지원한다.
- **numpy 반환, torch 텐서 아님.** 좌표는 데이터셋 원본 dtype(대개 float32) 그대로 준다.
