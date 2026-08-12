/*
 * SimpleModel — v2/model.py 의 `SimpleModel` 순수 JS 재구현 (전 데이터셋 학습본 fulltrain_best.pt).
 *
 * 가중치는 weights_simple.json(매니페스트 + Etot_table) + weights_simple.bin(float32 블롭).
 * 계산은 전부 float64(JS 기본 수치형)로 한다 — PyTorch float64와 1e-12 수준으로 일치한다.
 *
 * 에너지 = Σ Etot_table[Z]  (고립원자 총에너지, 수십~수백 Ha)  +  신경망 잔차(결합 몫, 수 Ha).
 * 즉 물이면 약 -76 Ha 다. 구모델(SimpleNNP, nnp.js)의 -0.4 Ha 와 스케일이 완전히 다르다.
 *
 * 힘 F = -∂E/∂r 은 **해석적 역전파**로 구한다. 좌표는 오직 어텐션 bias log(d_ij) 로만 들어가므로
 *   ∂E/∂r_k = Σ_{j≠k} (G_kj + G_jk) · (r_k - r_j)/d_kj²,   G_ij = ∂E/∂log(d_ij)
 * 이고, G는 energy_head → A3 → A2 → A1 순서로 한 번 역전파하면 나온다.
 * (유한차분이면 스텝당 forward 6N+1번 = 원자 10개에 61번 → 인터랙티브 불가. 검증용으로만
 *  energyAndForcesFD() 를 남겨 뒀다.)
 *
 * 좌표 Å, 에너지 Hartree. 원자번호 0 = 패딩(단일 분자 배치라 실제로는 안 쓰임).
 * 지원 원소: 학습 데이터(QM9x + ANI-2x)에 나온 H·C·N·O·F·S·Cl. 임베딩 표는 Z=92까지 있지만
 * 나머지는 학습되지 않은 초기값이라 의미 없는 값을 낸다.
 */
(function (global) {
  "use strict";

  class SimpleModel {
    /** manifest: weights_simple.json 파싱 결과, buffer: weights_simple.bin ArrayBuffer */
    constructor(manifest, buffer) {
      const cfg = manifest.config;
      this.cfg = cfg;
      this.H = cfg.atten_heads;        // 16
      this.DQK = cfg.head_qk_dim;      // 16
      this.DV = cfg.head_v_dim;        // 8
      this.D = cfg.inner_dim;          // 128
      this.ATT = cfg.atten_dim;        // 256
      this.NP = cfg.number_propo;      // 2 (원자번호 비례 head 수)
      this.FF = cfg.ffn_dim;           // 512
      this.scale = 1 / Math.sqrt(this.DQK);

      const f32 = new Float32Array(buffer);
      if (f32.length !== manifest.count) {
        throw new Error(`weights_simple.bin 크기 불일치: ${f32.length} != ${manifest.count}`);
      }
      const T = {};
      for (const t of manifest.tensors) {
        T[t.name] = Float64Array.from(f32.subarray(t.offset, t.offset + t.size));
      }
      this.T = T;
      this.lsd = T["log_self_distance"][0];              // 대각(자기 자신) attention bias
      this.Etot = Float64Array.from(manifest.Etot_table); // float64 그대로
      this._cache = null;  // 원자 배열이 그대로면 A1 Q/K/V·A1 score를 재사용
      this._ws = null;     // 원자 수별 작업 버퍼
    }

    // ---------------------------------------------------------------- 버퍼
    _work(N) {
      if (this._ws && this._ws.N === N) return this._ws;
      const { H, DQK, DV, D, FF } = this;
      const qkv = () => ({ q: new Float64Array(H * N * DQK), k: new Float64Array(H * N * DQK),
                           v: new Float64Array(H * N * DV) });
      this._ws = {
        N,
        L: new Float64Array(N * N), d2: new Float64Array(N * N),
        row: new Float64Array(N), dp: new Float64Array(N),
        A: [qkv(), qkv(), qkv()],                                   // 층별 Q/K/V
        P: [0, 1, 2].map(() => new Float64Array(H * N * N)),        // softmax 확률
        O: [0, 1, 2].map(() => new Float64Array(N * D)),            // attention 출력
        X: [0, 1, 2].map(() => new Float64Array(N * D)),            // FFN 출력
        G: [0, 1, 2].map(() => new Float64Array(N * FF)),           // FFN gate 선형값
        U: [0, 1, 2].map(() => new Float64Array(N * FF)),           // FFN up 선형값
        gX: new Float64Array(N * D), gO: new Float64Array(N * D),
        gq: new Float64Array(H * N * DQK), gk: new Float64Array(H * N * DQK),
        gv: new Float64Array(H * N * DV),
        da: new Float64Array(FF), dg: new Float64Array(FF), du: new Float64Array(FF),
        dy: new Float64Array(2 * this.ATT), dyv: new Float64Array(D),
        dL: new Float64Array(N * N),
      };
      return this._ws;
    }

    // ------------------------------------------------- 거리 / 어텐션 bias
    // model.py: 대각 rel_sq를 sqrt 전에 1로 채우고, bias는 대각만 학습 파라미터로 대체
    _distances(coords, N, w) {
      const { L, d2 } = w;
      for (let i = 0; i < N; i++) {
        L[i * N + i] = this.lsd;
        d2[i * N + i] = 1;
        for (let j = i + 1; j < N; j++) {
          const dx = coords[i][0] - coords[j][0];
          const dy = coords[i][1] - coords[j][1];
          const dz = coords[i][2] - coords[j][2];
          const s = dx * dx + dy * dy + dz * dz;
          const lg = 0.5 * Math.log(s);
          L[i * N + j] = lg; L[j * N + i] = lg;
          d2[i * N + j] = s; d2[j * N + i] = s;
        }
      }
    }

    // ------------------------------------------------------------ A1 Q/K/V
    // 원자번호에만 의존 → 완화 중에는 매 스텝 같다. 캐시.
    _qkvA1(numbers, N, w) {
      const key = numbers.join(",");
      if (this._cache && this._cache.key === key && this._cache.N === N) return;
      const { H, DQK, DV, NP, T } = this;
      const { q, k, v } = w.A[0];
      const wqk = T["A1_WQK.weight"], bqk = T["A1_WQK.bias"];
      const wv = T["A1_WV.weight"], bv = T["A1_WV.bias"];
      const eq = T["embq.weight"], ek = T["embk.weight"], ev = T["embv.weight"];
      const EQ = this.ATT - NP * DQK, EV = this.D - NP * DV;   // 임베딩 폭 (224, 112)
      for (let n = 0; n < N; n++) {
        const z = numbers[n];
        // 비례 head: 원자번호 스칼라의 선형변환. y[h*2D+d]=Q, y[h*2D+D+d]=K
        for (let h = 0; h < NP; h++) {
          const base = h * 2 * DQK;
          for (let d = 0; d < DQK; d++) {
            q[(h * N + n) * DQK + d] = wqk[base + d] * z + bqk[base + d];
            k[(h * N + n) * DQK + d] = wqk[base + DQK + d] * z + bqk[base + DQK + d];
          }
          for (let d = 0; d < DV; d++) {
            v[(h * N + n) * DV + d] = wv[h * DV + d] * z + bv[h * DV + d];
          }
        }
        // 비비례 head: 임베딩이 Q/K/V. head 축(dim=1)으로 뒤에 붙는다
        for (let h = NP; h < H; h++) {
          const e = h - NP;
          for (let d = 0; d < DQK; d++) {
            q[(h * N + n) * DQK + d] = eq[z * EQ + e * DQK + d];
            k[(h * N + n) * DQK + d] = ek[z * EQ + e * DQK + d];
          }
          for (let d = 0; d < DV; d++) {
            v[(h * N + n) * DV + d] = ev[z * EV + e * DV + d];
          }
        }
      }
      this._cache = { key, N };
    }

    // ------------------------------------ A2/A3: 이전 층 출력에서 Q/K/V 생성
    _qkvFrom(x, N, w, layer, prefix) {
      const { H, DQK, DV, D, ATT, T } = this;
      const { q, k, v } = w.A[layer];
      const wqk = T[`${prefix}_WQK.weight`], bqk = T[`${prefix}_WQK.bias`];
      const wv = T[`${prefix}_WV.weight`], bv = T[`${prefix}_WV.bias`];
      for (let n = 0; n < N; n++) {
        const xb = n * D;
        for (let h = 0; h < H; h++) {
          const rowQ = (h * 2 * DQK) * D, qb = (h * N + n) * DQK;
          for (let d = 0; d < DQK; d++) {
            let sq = bqk[h * 2 * DQK + d], sk = bqk[h * 2 * DQK + DQK + d];
            const rq = rowQ + d * D, rk = rowQ + (DQK + d) * D;
            for (let c = 0; c < D; c++) { sq += wqk[rq + c] * x[xb + c]; sk += wqk[rk + c] * x[xb + c]; }
            q[qb + d] = sq; k[qb + d] = sk;
          }
          const vb = (h * N + n) * DV;
          for (let d = 0; d < DV; d++) {
            let s = bv[h * DV + d];
            const rv = (h * DV + d) * D;
            for (let c = 0; c < D; c++) s += wv[rv + c] * x[xb + c];
            v[vb + d] = s;
          }
        }
      }
      void ATT;
    }

    // ------------------------------------------------------------- 어텐션
    // scores = q·k/√DQK − log(거리), softmax 후 V 가중합. 출력 [n][h*DV+d]
    _attend(N, w, layer) {
      const { H, DQK, DV, D, scale } = this;
      const { q, k, v } = w.A[layer];
      const L = w.L, P = w.P[layer], out = w.O[layer], row = w.row;
      out.fill(0);
      for (let h = 0; h < H; h++) {
        for (let i = 0; i < N; i++) {
          const qb = (h * N + i) * DQK;
          let m = -Infinity;
          for (let j = 0; j < N; j++) {
            const kb = (h * N + j) * DQK;
            let dot = 0;
            for (let d = 0; d < DQK; d++) dot += q[qb + d] * k[kb + d];
            const s = dot * scale - L[i * N + j];
            row[j] = s;
            if (s > m) m = s;
          }
          let sum = 0;
          for (let j = 0; j < N; j++) { const e = Math.exp(row[j] - m); row[j] = e; sum += e; }
          const inv = 1 / sum;
          const pb = (h * N + i) * N, ob = i * D + h * DV;
          for (let j = 0; j < N; j++) {
            const p = row[j] * inv;
            P[pb + j] = p;
            const vb = (h * N + j) * DV;
            for (let d = 0; d < DV; d++) out[ob + d] += p * v[vb + d];
          }
        }
      }
    }

    // --------------------------------------------------------- SwiGLU FFN
    _ffn(x, N, w, layer, prefix) {
      const { D, FF, T } = this;
      const wg = T[`${prefix}_SwiGLUFFN.w_gate.weight`];
      const wu = T[`${prefix}_SwiGLUFFN.w_up.weight`];
      const wd = T[`${prefix}_SwiGLUFFN.w_down.weight`];
      const G = w.G[layer], U = w.U[layer], out = w.X[layer], a = w.da;
      for (let n = 0; n < N; n++) {
        const xb = n * D, gb = n * FF;
        for (let f = 0; f < FF; f++) {
          const r = f * D;
          let sg = 0, su = 0;
          for (let c = 0; c < D; c++) { const xv = x[xb + c]; sg += wg[r + c] * xv; su += wu[r + c] * xv; }
          G[gb + f] = sg; U[gb + f] = su;
          a[f] = (sg / (1 + Math.exp(-sg))) * su;   // silu(gate) * up
        }
        for (let c = 0; c < D; c++) {
          const r = c * FF;
          let s = 0;
          for (let f = 0; f < FF; f++) s += wd[r + f] * a[f];
          out[xb + c] = s;
        }
      }
    }

    // --------------------------------------------------------------- forward
    /** numbers: int[N], coords: number[N][3] → 총에너지(Hartree) */
    energy(numbers, coords) {
      const N = numbers.length;
      const w = this._work(N);
      this._forward(numbers, coords, N, w);
      return w.E;
    }

    _forward(numbers, coords, N, w) {
      const { D, T } = this;
      this._distances(coords, N, w);
      this._qkvA1(numbers, N, w);
      this._attend(N, w, 0);
      this._ffn(w.O[0], N, w, 0, "A1");
      this._qkvFrom(w.X[0], N, w, 1, "A2");
      this._attend(N, w, 1);
      this._ffn(w.O[1], N, w, 1, "A2");
      this._qkvFrom(w.X[1], N, w, 2, "A3");
      this._attend(N, w, 2);
      this._ffn(w.O[2], N, w, 2, "A3");

      const wh = T["energy_head.weight"], bh = T["energy_head.bias"][0];
      const x = w.X[2];
      let nn = 0, iso = 0;
      for (let n = 0; n < N; n++) {
        let s = bh;
        for (let c = 0; c < D; c++) s += wh[c] * x[n * D + c];
        nn += s;
        iso += this.Etot[numbers[n]];
      }
      w.E = iso + nn;
    }

    // -------------------------------------------------------------- backward
    // gy(FFN 출력 기울기) → FFN 입력 기울기(gx에 누적)
    _ffnBack(N, w, layer, prefix, gy, gx) {
      const { D, FF, T } = this;
      const wg = T[`${prefix}_SwiGLUFFN.w_gate.weight`];
      const wu = T[`${prefix}_SwiGLUFFN.w_up.weight`];
      const wd = T[`${prefix}_SwiGLUFFN.w_down.weight`];
      const G = w.G[layer], U = w.U[layer], da = w.da, dg = w.dg, du = w.du;
      gx.fill(0);
      for (let n = 0; n < N; n++) {
        const xb = n * D, gb = n * FF;
        da.fill(0);
        for (let c = 0; c < D; c++) {
          const g = gy[xb + c];
          if (g === 0) continue;
          const r = c * FF;
          for (let f = 0; f < FF; f++) da[f] += g * wd[r + f];
        }
        for (let f = 0; f < FF; f++) {
          const gv = G[gb + f], uv = U[gb + f];
          const sg = 1 / (1 + Math.exp(-gv)), sl = gv * sg;   // sigmoid, silu
          du[f] = da[f] * sl;
          dg[f] = da[f] * uv * (sg + sl * (1 - sg));          // silu'(g)
        }
        for (let f = 0; f < FF; f++) {
          const a = dg[f], b = du[f];
          if (a === 0 && b === 0) continue;
          const r = f * D;
          for (let c = 0; c < D; c++) gx[xb + c] += a * wg[r + c] + b * wu[r + c];
        }
      }
    }

    // go(어텐션 출력 기울기) → dL 누적, 그리고 (wantQKV면) Q/K/V 기울기
    _attendBack(N, w, layer, go, wantQKV) {
      const { H, DQK, DV, D, scale } = this;
      const { q, k, v } = w.A[layer];
      const P = w.P[layer], dL = w.dL, dp = w.dp;
      const gq = w.gq, gk = w.gk, gv = w.gv;
      if (wantQKV) { gq.fill(0); gk.fill(0); gv.fill(0); }
      for (let h = 0; h < H; h++) {
        for (let i = 0; i < N; i++) {
          const pb = (h * N + i) * N, ob = i * D + h * DV, qb = (h * N + i) * DQK;
          let dot = 0;
          for (let j = 0; j < N; j++) {
            const p = P[pb + j], vb = (h * N + j) * DV;
            let s = 0;
            if (wantQKV) {
              for (let d = 0; d < DV; d++) { const g = go[ob + d]; s += g * v[vb + d]; gv[vb + d] += p * g; }
            } else {
              for (let d = 0; d < DV; d++) s += go[ob + d] * v[vb + d];
            }
            dp[j] = s;
            dot += p * s;
          }
          for (let j = 0; j < N; j++) {
            const ds = P[pb + j] * (dp[j] - dot);
            if (ds === 0) continue;
            // scores = q·k*scale - L → ∂E/∂L = -ds. 대각은 학습 파라미터(log_self_distance)라 제외
            if (i !== j) dL[i * N + j] -= ds;
            if (wantQKV) {
              const kb = (h * N + j) * DQK, sc = ds * scale;
              for (let d = 0; d < DQK; d++) { gq[qb + d] += sc * k[kb + d]; gk[kb + d] += sc * q[qb + d]; }
            }
          }
        }
      }
    }

    // Q/K/V 기울기 → 그 층 입력(x) 기울기
    _qkvBack(N, w, prefix, gx) {
      const { H, DQK, DV, D, ATT, T } = this;
      const wqk = T[`${prefix}_WQK.weight`], wv = T[`${prefix}_WV.weight`];
      const gq = w.gq, gk = w.gk, gv = w.gv, dy = w.dy, dyv = w.dyv;
      gx.fill(0);
      for (let n = 0; n < N; n++) {
        for (let h = 0; h < H; h++) {
          const qb = (h * N + n) * DQK, base = h * 2 * DQK;
          for (let d = 0; d < DQK; d++) { dy[base + d] = gq[qb + d]; dy[base + DQK + d] = gk[qb + d]; }
          const vb = (h * N + n) * DV;
          for (let d = 0; d < DV; d++) dyv[h * DV + d] = gv[vb + d];
        }
        const xb = n * D;
        for (let o = 0; o < 2 * ATT; o++) {
          const g = dy[o];
          if (g === 0) continue;
          const r = o * D;
          for (let c = 0; c < D; c++) gx[xb + c] += g * wqk[r + c];
        }
        for (let o = 0; o < D; o++) {
          const g = dyv[o];
          if (g === 0) continue;
          const r = o * D;
          for (let c = 0; c < D; c++) gx[xb + c] += g * wv[r + c];
        }
      }
    }

    /**
     * E 와 F = -∂E/∂r (해석적). numbers: int[N], coords: number[N][3]
     * 반환 { energy, forces: number[N][3] }
     */
    energyAndForces(numbers, coords) {
      const { D, T } = this;
      const N = numbers.length;
      const w = this._work(N);
      this._forward(numbers, coords, N, w);

      // energy_head → A3 FFN 입력(=A3 attention 출력) 기울기
      const wh = T["energy_head.weight"];
      const gy = w.gX;
      for (let n = 0; n < N; n++) for (let c = 0; c < D; c++) gy[n * D + c] = wh[c];
      w.dL.fill(0);

      this._ffnBack(N, w, 2, "A3", gy, w.gO);
      this._attendBack(N, w, 2, w.gO, true);
      this._qkvBack(N, w, "A3", w.gX);          // → A2 FFN 출력 기울기

      this._ffnBack(N, w, 1, "A2", w.gX, w.gO);
      this._attendBack(N, w, 1, w.gO, true);
      this._qkvBack(N, w, "A2", w.gX);          // → A1 FFN 출력 기울기

      this._ffnBack(N, w, 0, "A1", w.gX, w.gO);
      this._attendBack(N, w, 0, w.gO, false);   // A1 Q/K/V는 원자번호에만 의존 → dL만 필요

      // G = ∂E/∂log(d_ij) → 힘. ∂log d_ij/∂r_k = (r_k - r_j)/d² (i=k일 때)
      const dL = w.dL, d2 = w.d2;
      const forces = [];
      for (let i = 0; i < N; i++) {
        let fx = 0, fy = 0, fz = 0;
        for (let j = 0; j < N; j++) {
          if (i === j) continue;
          const g = (dL[i * N + j] + dL[j * N + i]) / d2[i * N + j];
          fx -= g * (coords[i][0] - coords[j][0]);
          fy -= g * (coords[i][1] - coords[j][1]);
          fz -= g * (coords[i][2] - coords[j][2]);
        }
        forces.push([fx, fy, fz]);
      }
      return { energy: w.E, forces };
    }

    /** 검증용: 중심 유한차분 힘 (스텝당 forward 6N+1번 — 느리다) */
    energyAndForcesFD(numbers, coords, eps = 1e-4) {
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

  /** 브라우저용: weights_simple.json + weights_simple.bin 을 받아 모델을 만든다 */
  SimpleModel.load = async function (base = "") {
    const p = base ? base.replace(/\/$/, "") + "/" : "";
    const [mres, bres] = await Promise.all([fetch(p + "weights_simple.json"), fetch(p + "weights_simple.bin")]);
    if (!mres.ok) throw new Error("weights_simple.json " + mres.status);
    if (!bres.ok) throw new Error("weights_simple.bin " + bres.status);
    return new SimpleModel(await mres.json(), await bres.arrayBuffer());
  };

  const api = { SimpleModel };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.SimpleNNP = api;
})(typeof window !== "undefined" ? window : globalThis);
