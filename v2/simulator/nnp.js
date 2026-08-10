/*
 * SimpleNNP — v2/model.py 의 순수 JS 재구현.
 *
 * v2/nnp.pt 의 가중치(weights.json)를 읽어 브라우저에서 직접 에너지를 계산한다.
 * 힘은 JS에 autograd가 없어 F = -∂E/∂r 을 중심 유한차분으로 구한다(원자 수가
 * 적어 스텝당 forward 6N번이면 충분). PyTorch 원본과 수치가 맞는지는 nnp.test.js 로 검증.
 *
 * 좌표 Å, 에너지 Hartree. 원자번호 0 = 패딩(단일 분자 배치라 실제로는 안 쓰임).
 * 지원 원소: H(1)~F(9) — 모델 임베딩이 MAX_Z=10 까지만 학습됨.
 */
(function (global) {
  "use strict";

  const silu = (x) => x / (1 + Math.exp(-x));

  // y[o] = Σ_i W[o][i]·x[i] + b[o]   (W: [out][in])
  function linear(x, W, b) {
    const out = new Array(W.length);
    for (let o = 0; o < W.length; o++) {
      const row = W[o];
      let s = b[o];
      for (let i = 0; i < row.length; i++) s += row[i] * x[i];
      out[o] = s;
    }
    return out;
  }

  function mlp(x, w0, b0, w2, b2) {
    const h = linear(x, w0, b0);
    for (let i = 0; i < h.length; i++) h[i] = silu(h[i]);
    return linear(h, w2, b2);
  }

  class SimpleNNP {
    constructor(weights) {
      this.w = weights;
      this.cfg = weights.config;
    }

    /** numbers: int[N], coords: number[N][3] → 에너지(Hartree, scalar) */
    energy(numbers, coords) {
      const w = this.w;
      const { hidden, n_rbf, cutoff, gamma, n_blocks } = this.cfg;
      const centers = w.centers;
      const N = numbers.length;

      const mask = numbers.map((z) => z > 0);

      // 거리·게이트·RBF (i,j 쌍)
      const dist = [], gate = [], rbf = [];
      for (let i = 0; i < N; i++) {
        dist.push(new Array(N));
        gate.push(new Array(N));
        rbf.push(new Array(N));
        for (let j = 0; j < N; j++) {
          const dx = coords[i][0] - coords[j][0];
          const dy = coords[i][1] - coords[j][1];
          const dz = coords[i][2] - coords[j][2];
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz + 1e-12);
          dist[i][j] = d;
          const pair = mask[i] && mask[j] && i !== j && d < cutoff;
          gate[i][j] = pair ? 0.5 * (Math.cos((Math.PI * d) / cutoff) + 1) : 0;
          if (pair) {
            const r = new Array(n_rbf);
            for (let k = 0; k < n_rbf; k++) {
              const t = d - centers[k];
              r[k] = Math.exp(-gamma * t * t);
            }
            rbf[i][j] = r;
          } else {
            rbf[i][j] = null; // gate=0 이라 어차피 안 쓰임
          }
        }
      }

      // 임베딩
      let h = numbers.map((z) => w.embedding[z].slice());

      // 메시지 패싱 블록
      for (let b = 0; b < n_blocks; b++) {
        const filt = w.filters[b], upd = w.updates[b];
        const newH = [];
        for (let i = 0; i < N; i++) {
          const msg = new Array(hidden).fill(0);
          for (let j = 0; j < N; j++) {
            const g = gate[i][j];
            if (g === 0) continue;
            const wf = mlp(rbf[i][j], filt.w0, filt.b0, filt.w2, filt.b2);
            const hj = h[j];
            for (let c = 0; c < hidden; c++) msg[c] += wf[c] * g * hj[c];
          }
          const du = mlp(msg, upd.w0, upd.b0, upd.w2, upd.b2);
          const hi = h[i], row = new Array(hidden);
          for (let c = 0; c < hidden; c++) row[c] = hi[c] + du[c];
          newH.push(row);
        }
        h = newH;
      }

      // readout → 원자별 에너지 합
      let E = 0;
      const ro = w.readout;
      for (let i = 0; i < N; i++) {
        if (!mask[i]) continue;
        E += mlp(h[i], ro.w0, ro.b0, ro.w2, ro.b2)[0];
      }
      return E;
    }

    /**
     * F = -∂E/∂r 를 중심 유한차분으로. moveMask[i] 가 false 인 원자(z 고정 등)는 건너뜀.
     * 반환: { energy, forces: number[N][3] }
     */
    energyAndForces(numbers, coords, eps = 1e-3) {
      const N = numbers.length;
      const E0 = this.energy(numbers, coords);
      const forces = coords.map(() => [0, 0, 0]);
      for (let i = 0; i < N; i++) {
        for (let d = 0; d < 3; d++) {
          const orig = coords[i][d];
          coords[i][d] = orig + eps;
          const ep = this.energy(numbers, coords);
          coords[i][d] = orig - eps;
          const em = this.energy(numbers, coords);
          coords[i][d] = orig;
          forces[i][d] = -(ep - em) / (2 * eps);
        }
      }
      return { energy: E0, forces };
    }
  }

  const api = { SimpleNNP };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.NNP = api;
})(typeof window !== "undefined" ? window : globalThis);
