/*
 * View3D — 분자 3D 뷰어 (canvas 2D + painter's algorithm).
 *
 * 외부 라이브러리 없이 직접 원근 투영한다. 원자가 30개 이하라 WebGL이 필요 없고,
 * 매 프레임 (투영 → 깊이 정렬 → 뒤에서 앞으로 그리기)만 하면 충분하다.
 *
 * 좌표계: world 는 Å 단위 오른손 좌표. 카메라는 target(뷰 중심)을 도는 orbit 카메라로
 * (yaw, pitch, dist) 세 값만 갖는다. 화면 y는 아래로 증가하므로 투영에서 부호를 뒤집는다.
 *
 * 사용법:
 *   const v = View3D.create(canvas, { getAtoms, colors, canPlace, canDragAtoms, onPlace, onAtomDrag });
 *   v.setEnabled(true);   // 렌더 루프는 항상 돌고, 꺼져 있으면 그리지 않는다
 *   v.frameAll();         // 현재 원자들이 화면에 들어오도록 target/dist 재설정
 */
(function (global) {
  "use strict";

  // ---- 상수 ----
  const FOV        = (45 * Math.PI) / 180; // 세로 화각
  const NEAR       = 0.05;                 // 근평면(Å). 이보다 가까운 정점은 잘라낸다
  const DIST_MIN   = 2.0;                  // 줌 인 한계(Å)
  const DIST_MAX   = 40;                   // 줌 아웃 한계(Å)
  const PITCH_LIM  = (85 * Math.PI) / 180; // 짐벌락 방지
  const ROT_SPEED  = 0.008;                // 회전 감도(rad per px)
  const ZOOM_SPEED = 0.0016;               // 줌 감도(per wheel delta)
  const CLICK_SLOP = 4;                    // 이보다 적게 움직이면 드래그가 아니라 클릭
  const BOND_MAX   = 1.8;                  // 결합선 기준 거리(Å) — 2D 모드와 동일
  const BOND_W     = 0.055;                // 결합선 굵기(Å 기준, 원근으로 스케일)
  const RADII      = { H: 0.30, C: 0.42, N: 0.40, O: 0.38, F: 0.36 };
  const R_DEFAULT  = 0.40;                 // 미등록 원소 반경(Å)
  const LABEL_MIN  = 9;                    // 원 반경(px)이 이보다 작으면 라벨 생략
  const FADE_MAX   = 0.5;                  // 깊이에 따른 최대 흐려짐(0=없음, 1=배경색)
  const GRID_DROP  = 2.6;                  // 바닥 그리드를 target 아래 몇 Å에 둘지
  const GRID_HALF  = 4.0;                  // 그리드 반폭(Å)
  const GRID_STEP  = 0.8;                  // 그리드 간격(Å)
  const AXIS_LEN   = 1.0;                  // 축 표시 길이(Å)
  const BG         = [244, 243, 238];      // --paper. 깊이 감쇠 때 섞을 밝은 배경색
  const GRID_COLOR = "rgba(31,59,255,.10)";
  const BOND_COLOR = "#9aa1ad";
  const AXIS_COLORS = ["rgba(255,90,54,.45)", "rgba(90,97,107,.40)", "rgba(31,59,255,.40)"]; // x, y, z

  // ---- 벡터 유틸 ----
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);

  // hex 색을 밝은 배경 쪽으로 t(0~1)만큼 섞는다 — 멀수록 옅게 보이도록
  function fade(hex, t) {
    const h = hex.replace("#", "");
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return "rgb(" + Math.round(r + (BG[0] - r) * t) + "," +
                    Math.round(g + (BG[1] - g) * t) + "," +
                    Math.round(b + (BG[2] - b) * t) + ")";
  }

  function create(canvas, opts) {
    const o = Object.assign({
      getAtoms: () => [],       // → [{el, p:[x,y,z]}] (배열/좌표 모두 라이브 참조)
      colors: {},               // { H:{c,t}, ... } — index.html 의 ELEMENTS 재사용
      canPlace: () => false,    // 빈 공간 클릭을 "배치"로 볼지
      canDragAtoms: () => true, // 원자 드래그 허용 여부(완화 중에는 false)
      onPlace: () => {},        // (worldPos) 빈 공간 클릭
      onAtomDrag: () => {},     // (index, worldPos) 원자 드래그
    }, opts);

    const ctx = canvas.getContext("2d");
    const cam = { target: [0, 0, 0], dist: 8, yaw: 0.65, pitch: 0.32 };
    let W = 1, H = 1, focal = 1;
    let enabled = false, raf = 0;
    let eye = [0, 0, 0], fwd = [0, 0, -1], right = [1, 0, 0], up = [0, 1, 0];

    // ---- 카메라 ----
    function resize() {
      const dpr = global.devicePixelRatio || 1;
      W = canvas.clientWidth || 1;
      H = canvas.clientHeight || 1;
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      focal = (H / 2) / Math.tan(FOV / 2);
    }

    function updateBasis() {
      const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
      const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
      const off = [cam.dist * cp * sy, cam.dist * sp, cam.dist * cp * cy];
      eye = add(cam.target, off);
      fwd = mul(off, -1 / cam.dist);   // target 쪽을 보는 단위벡터
      right = [cy, 0, -sy];            // = normalize(cross(fwd, worldUp))
      up = [-sp * sy, cp, -sp * cy];   // = cross(right, fwd)
    }

    // world → 카메라 좌표 [수평, 수직, 깊이]
    const camOf = (p) => {
      const v = sub(p, eye);
      return [dot(v, right), dot(v, up), dot(v, fwd)];
    };
    // 카메라 좌표 → 화면 px
    const toScreen = (c) => [W / 2 + (focal * c[0]) / c[2], H / 2 - (focal * c[1]) / c[2]];

    // 화면 px + 깊이 → world (그 깊이의 카메라 평면 위 점)
    function unproject(sx, sy, depth) {
      const cx = ((sx - W / 2) * depth) / focal;
      const cy = (-(sy - H / 2) * depth) / focal;
      return add(add(eye, mul(fwd, depth)), add(mul(right, cx), mul(up, cy)));
    }
    // 뷰 타깃을 지나는 정면 평면 위의 점
    const screenToTargetPlane = (sx, sy) => unproject(sx, sy, cam.dist);

    // 근평면 기준 선분 클리핑(카메라 좌표에서)
    function clipSeg(a, b) {
      if (a[2] < NEAR && b[2] < NEAR) return null;
      const lerp = (p, q, t) => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t];
      if (a[2] < NEAR) a = lerp(a, b, (NEAR - a[2]) / (b[2] - a[2]));
      else if (b[2] < NEAR) b = lerp(b, a, (NEAR - b[2]) / (a[2] - b[2]));
      return [a, b];
    }

    // 깊이 → 흐려짐 정도(0=선명). 카메라 거리를 기준으로 정규화
    const depthFade = (z) => clamp((z - (cam.dist - 1.5)) / (cam.dist * 1.1), 0, 1) * FADE_MAX;

    // ---- 그리기 ----
    function drawSeg(pa, pb, color, width) {
      const seg = clipSeg(camOf(pa), camOf(pb));
      if (!seg) return;
      const s0 = toScreen(seg[0]), s1 = toScreen(seg[1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(s0[0], s0[1]);
      ctx.lineTo(s1[0], s1[1]);
      ctx.stroke();
    }

    // 바닥 그리드 + 축 표시 — 공간감을 주되 밝은 배경을 해치지 않게 아주 옅게
    function drawGrid() {
      const gy = cam.target[1] - GRID_DROP;
      const gx = Math.round(cam.target[0] / GRID_STEP) * GRID_STEP;
      const gz = Math.round(cam.target[2] / GRID_STEP) * GRID_STEP;
      ctx.lineCap = "butt";
      for (let k = -GRID_HALF; k <= GRID_HALF + 1e-9; k += GRID_STEP) {
        drawSeg([gx + k, gy, gz - GRID_HALF], [gx + k, gy, gz + GRID_HALF], GRID_COLOR, 1);
        drawSeg([gx - GRID_HALF, gy, gz + k], [gx + GRID_HALF, gy, gz + k], GRID_COLOR, 1);
      }
      const org = [gx, gy, gz];
      drawSeg(org, add(org, [AXIS_LEN, 0, 0]), AXIS_COLORS[0], 2);
      drawSeg(org, add(org, [0, AXIS_LEN, 0]), AXIS_COLORS[1], 2);
      drawSeg(org, add(org, [0, 0, AXIS_LEN]), AXIS_COLORS[2], 2);
      ctx.lineCap = "round";
    }

    function drawAtom(atom, c) {
      if (c[2] < NEAR) return;
      const s = toScreen(c);
      const rp = (focal * (RADII[atom.el] || R_DEFAULT)) / c[2];
      const sty = o.colors[atom.el] || { c: "#888888", t: "#fff" };
      const t = depthFade(c[2]);
      ctx.beginPath();
      ctx.arc(s[0], s[1], rp, 0, Math.PI * 2);
      ctx.fillStyle = fade(sty.c, t);
      ctx.fill();
      ctx.lineWidth = Math.max(1, rp * 0.1);
      ctx.strokeStyle = "rgba(0,0,0," + (0.24 * (1 - t)).toFixed(3) + ")";
      ctx.stroke();
      if (rp >= LABEL_MIN) {
        ctx.fillStyle = fade(sty.t.length === 7 ? sty.t : "#333333", t * 0.6);
        ctx.font = "800 " + Math.round(rp * 0.95) + "px Pretendard, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(atom.el, s[0], s[1] + rp * 0.04);
      }
    }

    function render() {
      raf = global.requestAnimationFrame(render);
      if (!enabled) return;
      if (canvas.clientWidth !== W || canvas.clientHeight !== H) resize();
      updateBasis();
      ctx.clearRect(0, 0, W, H);
      ctx.lineCap = "round";
      drawGrid();

      const atoms = o.getAtoms();
      const P = atoms.map((a) => camOf(a.p));

      // 원자와 결합선을 한 목록에 모아 깊이 내림차순(먼 것 먼저)으로 그린다
      const items = [];
      for (let i = 0; i < atoms.length; i++) {
        items.push({ atom: 1, i: i, z: P[i][2] });
        for (let j = i + 1; j < atoms.length; j++) {
          if (norm(sub(atoms[i].p, atoms[j].p)) < BOND_MAX)
            items.push({ atom: 0, i: i, j: j, z: (P[i][2] + P[j][2]) / 2 });
        }
      }
      items.sort((a, b) => b.z - a.z);

      for (const it of items) {
        if (it.atom) drawAtom(atoms[it.i], P[it.i]);
        else drawSeg(atoms[it.i].p, atoms[it.j].p, fade(BOND_COLOR, depthFade(it.z)),
                     clamp((focal * BOND_W) / Math.max(it.z, NEAR), 1, 9));
      }
    }

    // ---- 입력 ----
    function localXY(e) {
      const r = canvas.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    }

    // 화면에서 가장 앞에 있는(=깊이가 작은) 원자부터 검사
    function pick(sx, sy) {
      updateBasis();
      const atoms = o.getAtoms();
      const hits = [];
      for (let i = 0; i < atoms.length; i++) {
        const c = camOf(atoms[i].p);
        if (c[2] < NEAR) continue;
        const s = toScreen(c);
        const rp = (focal * (RADII[atoms[i].el] || R_DEFAULT)) / c[2];
        if (Math.hypot(sx - s[0], sy - s[1]) <= rp) hits.push({ i: i, z: c[2] });
      }
      if (!hits.length) return null;
      hits.sort((a, b) => a.z - b.z);
      return hits[0];
    }

    let drag = null; // {kind:"orbit"|"atom", x, y, moved, index, depth}

    canvas.addEventListener("pointerdown", (e) => {
      if (!enabled) return;
      e.preventDefault();
      canvas.setPointerCapture(e.pointerId);
      const [x, y] = localXY(e);
      const hit = o.canDragAtoms() ? pick(x, y) : null;
      drag = hit
        ? { kind: "atom", x: x, y: y, moved: 0, index: hit.i, depth: hit.z }
        : { kind: "orbit", x: x, y: y, moved: 0 };
      canvas.style.cursor = "grabbing";
    });

    canvas.addEventListener("pointermove", (e) => {
      if (!enabled) return;
      const [x, y] = localXY(e);
      if (!drag) {
        canvas.style.cursor = (o.canDragAtoms() && pick(x, y)) ? "move" : "grab";
        return;
      }
      const dx = x - drag.x, dy = y - drag.y;
      drag.moved += Math.abs(dx) + Math.abs(dy);
      drag.x = x; drag.y = y;
      if (drag.kind === "orbit") {
        cam.yaw -= dx * ROT_SPEED;
        cam.pitch = clamp(cam.pitch + dy * ROT_SPEED, -PITCH_LIM, PITCH_LIM);
      } else {
        updateBasis();
        o.onAtomDrag(drag.index, unproject(x, y, drag.depth));
      }
    });

    function endDrag(e) {
      if (!drag) return;
      const [x, y] = localXY(e);
      const wasClick = drag.moved < CLICK_SLOP;
      const kind = drag.kind;
      drag = null;
      canvas.style.cursor = "grab";
      if (wasClick && kind === "orbit" && o.canPlace()) {
        updateBasis();
        o.onPlace(screenToTargetPlane(x, y));
      }
    }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    canvas.addEventListener("wheel", (e) => {
      if (!enabled) return;
      e.preventDefault();
      cam.dist = clamp(cam.dist * Math.exp(e.deltaY * ZOOM_SPEED), DIST_MIN, DIST_MAX);
    }, { passive: false });

    // ---- 공개 API ----
    function frameAll() {
      const atoms = o.getAtoms();
      if (atoms.length) {
        const c = [0, 0, 0];
        for (const a of atoms) { c[0] += a.p[0]; c[1] += a.p[1]; c[2] += a.p[2]; }
        cam.target = mul(c, 1 / atoms.length);
        let rad = 0;
        for (const a of atoms) rad = Math.max(rad, norm(sub(a.p, cam.target)));
        cam.dist = clamp(rad / Math.tan(FOV / 2) + 2.6, 3.5, DIST_MAX);
      } else {
        cam.target = [0, 0, 0];
        cam.dist = 8;
      }
      updateBasis();
    }

    resize();
    updateBasis();
    raf = global.requestAnimationFrame(render);

    return {
      setEnabled(v) { enabled = !!v; if (enabled) resize(); },
      resize: resize,
      frameAll: frameAll,
      getTarget: () => cam.target.slice(),
      getForward: () => fwd.slice(),
      screenToTargetPlane: (sx, sy) => { updateBasis(); return screenToTargetPlane(sx, sy); },
      destroy() { global.cancelAnimationFrame(raf); },
    };
  }

  const api = { create: create };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else global.View3D = api;
})(typeof window !== "undefined" ? window : globalThis);
