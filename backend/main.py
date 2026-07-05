"""
NNP 추론 서빙 API (FastAPI) — 학습된 모델 연결 버전

- 기동 시 학습 모델(model/predict_forces.py)을 1회 로드해 GPU에 올림(재요청마다 재사용).
- POST /predict : 원자(원소 + 좌표 Å) → 총에너지(Hartree) + 원자별 힘(Hartree/Å)
- POST /step    : 힘 방향 최속강하 한 스텝(완화 데모용)
- GET  /health  : 상태 점검

⚠️ 단위: 좌표 Å, 에너지 Hartree, 힘 Hartree/Å. 힘 정의 F_i = -∂E/∂r_i.
⚠️ 힘은 autograd로 계산 → no_grad / half(fp16)로 감싸면 안 됨. fp32 유지.
   (원래 요구의 no_grad/fp16은 INFERENCE_CONTRACT.md 지침에 따라 제외함.)
⚠️ 지원 원소: H/C/N/O 4종만. 그 외는 422로 거절.

실행 (AWS EC2):
    pip install -r requirements.txt          # torch는 CUDA 빌드로 별도 설치
    uvicorn main:app --host 0.0.0.0 --port 8000
    # model/checkpoints/checkpoint_best.pt 가 있어야 함(없으면 /health 가 model_not_loaded).
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# model/ 폴더의 추론 코드를 import 경로에 추가
# (predict_forces.py, model.py, preprocess.py 가 그 안에 함께 있어야 함)
_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(_MODEL_DIR))
from predict_forces import load_model, predict_energy_forces  # noqa: E402

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
MAX_CONCURRENT_INFER = 4              # GPU 동시 추론 제한
SUPPORTED = {"H", "C", "N", "O"}      # 모델이 학습한 원소 4종
_SYM2Z = {"H": 1, "C": 6, "N": 7, "O": 8}  # 원소기호 → 원자번호

# GPU 동시 추론을 4개로 제한하는 세마포어 (Python 3.10+ : 모듈 레벨 생성 가능)
gpu_sem = asyncio.Semaphore(MAX_CONCURRENT_INFER)

# 로드된 (model, energy_ref, target_std) 튜플 — lifespan에서 1회 로드해 재사용
_ckpt = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 기동 시 체크포인트를 1회 로드(GPU로 이동). 실패해도 서버는 뜨고 /health에 드러남."""
    global _ckpt
    try:
        _ckpt = load_model()          # (model, ref, std)
    except Exception as e:
        _ckpt = None
        print(f"[startup] 모델 로드 실패: {e}", file=sys.stderr)
    yield
    _ckpt = None


# ----------------------------------------------------------------------
# 입출력 스키마 (pydantic) — HTTP 계약은 기존과 동일하게 유지
# ----------------------------------------------------------------------
class Atom(BaseModel):
    element: str = Field(..., description="원소 기호 (H/C/N/O 만)")
    x: float
    y: float
    z: float

    @field_validator("element")
    @classmethod
    def _supported(cls, v: str) -> str:
        v = v.strip().capitalize()    # 'h' → 'H', 'cl' → 'Cl'
        if v not in SUPPORTED:
            raise ValueError(f"지원하지 않는 원소: {v!r} (H/C/N/O 만 가능)")
        return v


class PredictRequest(BaseModel):
    atoms: list[Atom] = Field(..., min_length=1, description="분자를 구성하는 원자 목록")


class PredictResponse(BaseModel):
    energy: float = Field(..., description="총에너지 (Hartree)")
    forces: list[list[float]] = Field(..., description="원자별 힘 [[fx,fy,fz],...] (Hartree/Å)")
    n_atoms: int
    unit: str = "Hartree"             # 에너지 단위 (힘은 Hartree/Å)
    note: str = ""


class StepRequest(BaseModel):
    """MD 완화 한 스텝 요청 (인터랙티브 데모용). 좌표 단위는 Å."""
    atoms: list[Atom] = Field(..., min_length=1, description="현재 원자 배치")
    dt: float = Field(0.05, gt=0, description="스텝 크기")


class StepResponse(BaseModel):
    atoms: list[list[float]] = Field(..., description="다음 스텝 좌표 [[x,y,z],...] (Å)")
    energy: float
    note: str = ""


# ----------------------------------------------------------------------
# 추론 (블로킹) — run_in_threadpool 로 호출. autograd 사용이라 no_grad 금지.
# ----------------------------------------------------------------------
def _to_ZR(atoms: list[Atom]):
    Z = np.array([_SYM2Z[a.element] for a in atoms], dtype=np.int64)   # 원자번호
    R = np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float32)   # 좌표 Å
    return Z, R


def run_inference(req: PredictRequest) -> PredictResponse:
    Z, R = _to_ZR(req.atoms)
    res = predict_energy_forces(Z, R, ckpt=_ckpt)        # ckpt 재사용(재로딩 방지)
    return PredictResponse(
        energy=float(res["energy_hartree"]),             # Hartree
        forces=np.asarray(res["forces"], dtype=float).tolist(),  # Hartree/Å
        n_atoms=len(req.atoms),
        unit="Hartree",
        note="forces: Hartree/Å, F=-∂E/∂r",
    )


def run_step(req: StepRequest) -> StepResponse:
    Z, R = _to_ZR(req.atoms)
    res = predict_energy_forces(Z, R, ckpt=_ckpt)
    F = np.asarray(res["forces"], dtype=np.float32)      # (n,3) Hartree/Å
    step = np.clip(req.dt * F, -0.1, 0.1)                # 발산 방지: dt 작게 + 클리핑
    R_new = (R + step).tolist()                          # 힘 방향으로 이동 → 에너지 감소
    return StepResponse(
        atoms=R_new,
        energy=float(res["energy_hartree"]),
        note="최속강하 1스텝 (F=-∂E/∂r 방향)",
    )


# ----------------------------------------------------------------------
# FastAPI 앱
# ----------------------------------------------------------------------
app = FastAPI(title="NNP 추론 API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://34-123hs.github.io",   # 배포된 데모(GitHub Pages)
        "http://localhost:8000",        # 로컬 데모 테스트
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    import torch
    return {
        "status": "ok" if _ckpt is not None else "model_not_loaded",
        "model_loaded": _ckpt is not None,
        "cuda": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "supported_elements": sorted(SUPPORTED),
        "max_concurrent_infer": MAX_CONCURRENT_INFER,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    if _ckpt is None:
        raise HTTPException(503, "모델이 로드되지 않음 (model/checkpoints/checkpoint_best.pt 확인)")
    async with gpu_sem:               # GPU 동시 추론 4개로 제한
        return await run_in_threadpool(run_inference, req)


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest) -> StepResponse:
    if _ckpt is None:
        raise HTTPException(503, "모델이 로드되지 않음 (model/checkpoints/checkpoint_best.pt 확인)")
    async with gpu_sem:
        return await run_in_threadpool(run_step, req)
