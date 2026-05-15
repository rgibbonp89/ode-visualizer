import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';

// ============================================================================
// Math helpers
// ============================================================================

const matMul = (A, B) => [
  [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
  [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
];
const matVec = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
const transpose = (M) => [[M[0][0], M[1][0]], [M[0][1], M[1][1]]];
const rotMat = (theta) => {
  const c = Math.cos(theta), s = Math.sin(theta);
  return [[c, -s], [s, c]];
};

// Symmetric M from eigenvalues + rotation: M = R diag(l1,l2) R^T
const symMfromEig = (l1, l2, theta) => {
  const R = rotMat(theta);
  const D = [[l1, 0], [0, l2]];
  return matMul(matMul(R, D), transpose(R));
};

// Apply exp(-tM) to vector g, given M's eigendecomp (l1, l2, theta)
const applyExpDecay = (t, l1, l2, theta, g) => {
  const c = Math.cos(theta), s = Math.sin(theta);
  const u1 = c * g[0] + s * g[1];
  const u2 = -s * g[0] + c * g[1];
  const v1 = u1 * Math.exp(-t * l1);
  const v2 = u2 * Math.exp(-t * l2);
  return [c * v1 - s * v2, s * v1 + c * v2];
};

// exp(A) for symmetric 2x2 matrix A via eigendecomposition
const matExpSym = (A) => {
  const a = A[0][0], b = A[0][1], d = A[1][1];
  const tr = a + d;
  const halfTr = tr / 2;
  const det = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, halfTr * halfTr - det));
  const l1 = halfTr + disc;
  const l2 = halfTr - disc;
  let v1;
  if (Math.abs(b) > 1e-10) {
    const raw = [b, l1 - a];
    const n = Math.hypot(raw[0], raw[1]);
    v1 = [raw[0] / n, raw[1] / n];
  } else {
    v1 = a >= d ? [1, 0] : [0, 1];
  }
  const v2 = [-v1[1], v1[0]];
  const e1 = Math.exp(l1), e2 = Math.exp(l2);
  return [
    [e1 * v1[0] * v1[0] + e2 * v2[0] * v2[0], e1 * v1[0] * v1[1] + e2 * v2[0] * v2[1]],
    [e1 * v1[1] * v1[0] + e2 * v2[1] * v2[0], e1 * v1[1] * v1[1] + e2 * v2[1] * v2[1]],
  ];
};

// RK4 integration of dg/dt = -M(t) g
const integrateODE = (g0, getM, T, steps) => {
  const dt = T / steps;
  const traj = [{ t: 0, g: g0 }];
  let g = g0;
  const f = (t, gv) => {
    const M = getM(t);
    return [-(M[0][0] * gv[0] + M[0][1] * gv[1]), -(M[1][0] * gv[0] + M[1][1] * gv[1])];
  };
  for (let i = 0; i < steps; i++) {
    const t = i * dt;
    const k1 = f(t, g);
    const k2 = f(t + dt / 2, [g[0] + (dt / 2) * k1[0], g[1] + (dt / 2) * k1[1]]);
    const k3 = f(t + dt / 2, [g[0] + (dt / 2) * k2[0], g[1] + (dt / 2) * k2[1]]);
    const k4 = f(t + dt, [g[0] + dt * k3[0], g[1] + dt * k3[1]]);
    g = [
      g[0] + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
      g[1] + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    ];
    traj.push({ t: (i + 1) * dt, g });
  }
  return traj;
};

// Cumulative ∫_0^t M(τ) dτ via trapezoidal rule
const cumulativeIntM = (getM, T, steps) => {
  const dt = T / steps;
  const out = [{ t: 0, A: [[0, 0], [0, 0]] }];
  let A = [[0, 0], [0, 0]];
  let prevM = getM(0);
  for (let i = 1; i <= steps; i++) {
    const t = i * dt;
    const M = getM(t);
    A = [
      [A[0][0] + (dt * (prevM[0][0] + M[0][0])) / 2, A[0][1] + (dt * (prevM[0][1] + M[0][1])) / 2],
      [A[1][0] + (dt * (prevM[1][0] + M[1][0])) / 2, A[1][1] + (dt * (prevM[1][1] + M[1][1])) / 2],
    ];
    out.push({ t, A: [[A[0][0], A[0][1]], [A[1][0], A[1][1]]] });
    prevM = M;
  }
  return out;
};

// ============================================================================
// Palette
// ============================================================================

const C = {
  bg: '#FAF7F2',
  panel: '#FFFFFF',
  border: '#E5DDD0',
  text: '#1A1612',
  muted: '#8B847A',
  faint: '#BFB8AC',
  burgundy: '#8B2535',
  teal: '#1E7A7A',
  copper: '#C77B3F',
  ink: '#2A2520',
  naive: '#B8B8B8',
};

// ============================================================================
// UI atoms
// ============================================================================

const Slider = ({ label, value, onChange, min, max, step, fmt = (v) => v.toFixed(2), unit = '' }) => (
  <div className="flex flex-col gap-1">
    <div className="flex items-baseline justify-between text-xs">
      <span style={{ color: C.muted }} className="font-medium tracking-wide uppercase">{label}</span>
      <span style={{ color: C.ink, fontFamily: 'Fraunces, serif' }} className="tabular-nums italic">
        {fmt(value)}{unit}
      </span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full"
      style={{ accentColor: C.burgundy }}
    />
  </div>
);

const Tab = ({ active, onClick, children, sub }) => (
  <button
    onClick={onClick}
    className="px-4 py-2.5 text-left transition-all relative"
    style={{
      background: active ? C.ink : 'transparent',
      color: active ? C.bg : C.muted,
      borderBottom: active ? `2px solid ${C.copper}` : `2px solid transparent`,
    }}
  >
    <div className="text-xs uppercase tracking-widest font-semibold">{children}</div>
    {sub && (
      <div
        className="text-xs mt-0.5 italic"
        style={{ fontFamily: 'Fraunces, serif', color: active ? C.faint : C.muted }}
      >
        {sub}
      </div>
    )}
  </button>
);

const Toggle = ({ options, value, onChange }) => (
  <div className="inline-flex rounded-sm overflow-hidden" style={{ border: `1px solid ${C.border}` }}>
    {options.map((opt) => (
      <button
        key={opt.value}
        onClick={() => onChange(opt.value)}
        className="px-3 py-1.5 text-xs uppercase tracking-wider transition-colors"
        style={{
          background: value === opt.value ? C.ink : C.panel,
          color: value === opt.value ? C.bg : C.muted,
          fontWeight: value === opt.value ? 600 : 500,
        }}
      >
        {opt.label}
      </button>
    ))}
  </div>
);

const Eq = ({ children }) => (
  <div
    className="py-3 px-5 my-2 inline-block"
    style={{
      fontFamily: 'Fraunces, serif',
      fontStyle: 'italic',
      fontSize: '1.05rem',
      background: C.bg,
      borderLeft: `2px solid ${C.copper}`,
      color: C.ink,
    }}
  >
    {children}
  </div>
);

const Stat = ({ label, value, color = C.ink }) => (
  <div className="flex flex-col">
    <div className="text-xs uppercase tracking-wider" style={{ color: C.muted }}>{label}</div>
    <div
      className="tabular-nums italic"
      style={{ fontFamily: 'Fraunces, serif', fontSize: '1.1rem', color }}
    >
      {value}
    </div>
  </div>
);

// ============================================================================
// Plot primitives
// ============================================================================

const LinePlot = ({
  series, // array of { data: [{x,y}], color, dashed, width }
  width,
  height,
  xRange,
  yRange,
  xLabel,
  yLabel,
  markers = [], // { x, y, color, r, label }
  vlines = [], // { x, color, dashed, label }
  hlines = [], // { y, color, dashed, label }
  xTicks,
  yTicks,
  title,
}) => {
  const padL = 44, padR = 12, padT = title ? 26 : 12, padB = 30;
  const W = width - padL - padR;
  const H = height - padT - padB;
  const x2px = (x) => padL + ((x - xRange[0]) / (xRange[1] - xRange[0])) * W;
  const y2px = (y) => padT + (1 - (y - yRange[0]) / (yRange[1] - yRange[0])) * H;

  const xTickVals = xTicks || [xRange[0], (xRange[0] + xRange[1]) / 2, xRange[1]];
  const yTickVals = yTicks || [yRange[0], (yRange[0] + yRange[1]) / 2, yRange[1]];

  const pathFor = (data) =>
    data.length === 0
      ? ''
      : data.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x2px(p.x).toFixed(2)} ${y2px(p.y).toFixed(2)}`).join(' ');

  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      {title && (
        <text
          x={padL}
          y={14}
          style={{ fontFamily: 'Fraunces, serif', fontSize: 13, fontStyle: 'italic', fill: C.ink }}
        >
          {title}
        </text>
      )}
      {/* Axes */}
      <line x1={padL} y1={padT} x2={padL} y2={padT + H} stroke={C.faint} strokeWidth={1} />
      <line x1={padL} y1={padT + H} x2={padL + W} y2={padT + H} stroke={C.faint} strokeWidth={1} />

      {/* y ticks */}
      {yTickVals.map((v, i) => (
        <g key={`yt${i}`}>
          <line x1={padL - 4} y1={y2px(v)} x2={padL} y2={y2px(v)} stroke={C.faint} strokeWidth={1} />
          <text
            x={padL - 7}
            y={y2px(v) + 3}
            textAnchor="end"
            style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 9, fill: C.muted }}
          >
            {typeof v === 'number' ? v.toFixed(Math.abs(v) >= 10 ? 0 : 1) : v}
          </text>
        </g>
      ))}
      {/* x ticks */}
      {xTickVals.map((v, i) => (
        <g key={`xt${i}`}>
          <line x1={x2px(v)} y1={padT + H} x2={x2px(v)} y2={padT + H + 4} stroke={C.faint} strokeWidth={1} />
          <text
            x={x2px(v)}
            y={padT + H + 14}
            textAnchor="middle"
            style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 9, fill: C.muted }}
          >
            {typeof v === 'number' ? v.toFixed(Math.abs(v) < 1 && v !== 0 ? 2 : 1) : v}
          </text>
        </g>
      ))}

      {/* h-lines */}
      {hlines.map((l, i) => (
        <g key={`hl${i}`}>
          <line
            x1={padL}
            y1={y2px(l.y)}
            x2={padL + W}
            y2={y2px(l.y)}
            stroke={l.color || C.faint}
            strokeWidth={1}
            strokeDasharray={l.dashed ? '3 3' : 'none'}
            opacity={0.7}
          />
          {l.label && (
            <text
              x={padL + W - 4}
              y={y2px(l.y) - 3}
              textAnchor="end"
              style={{ fontFamily: 'Fraunces, serif', fontSize: 10, fontStyle: 'italic', fill: l.color || C.muted }}
            >
              {l.label}
            </text>
          )}
        </g>
      ))}
      {/* v-lines */}
      {vlines.map((l, i) => (
        <g key={`vl${i}`}>
          <line
            x1={x2px(l.x)}
            y1={padT}
            x2={x2px(l.x)}
            y2={padT + H}
            stroke={l.color || C.faint}
            strokeWidth={1}
            strokeDasharray={l.dashed ? '3 3' : 'none'}
            opacity={0.7}
          />
          {l.label && (
            <text
              x={x2px(l.x) + 4}
              y={padT + 10}
              style={{ fontFamily: 'Fraunces, serif', fontSize: 10, fontStyle: 'italic', fill: l.color || C.muted }}
            >
              {l.label}
            </text>
          )}
        </g>
      ))}

      {/* series */}
      {series.map((s, i) => (
        <path
          key={`s${i}`}
          d={pathFor(s.data)}
          stroke={s.color}
          fill="none"
          strokeWidth={s.width || 2}
          strokeDasharray={s.dashed ? '5 4' : 'none'}
          strokeLinejoin="round"
          strokeLinecap="round"
          opacity={s.opacity || 1}
        />
      ))}

      {/* markers */}
      {markers.map((m, i) => (
        <g key={`m${i}`}>
          <circle
            cx={x2px(m.x)}
            cy={y2px(m.y)}
            r={m.r || 4}
            fill={m.fill || m.color}
            stroke={m.stroke || m.color}
            strokeWidth={m.strokeWidth || 1.5}
          />
          {m.label && (
            <text
              x={x2px(m.x) + 8}
              y={y2px(m.y) + 4}
              style={{ fontFamily: 'Fraunces, serif', fontSize: 10, fontStyle: 'italic', fill: m.color }}
            >
              {m.label}
            </text>
          )}
        </g>
      ))}

      {/* labels */}
      {xLabel && (
        <text
          x={padL + W / 2}
          y={height - 4}
          textAnchor="middle"
          style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontSize: 11, fill: C.muted }}
        >
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          x={-padT - H / 2}
          y={12}
          transform="rotate(-90)"
          textAnchor="middle"
          style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontSize: 11, fill: C.muted }}
        >
          {yLabel}
        </text>
      )}
    </svg>
  );
};

// 2D phase-space plot, optionally with draggable initial point
const PhasePlot = ({
  size = 440,
  range = 3, // axis goes from -range to +range
  trajectories = [], // { data: [[x,y]], color, dashed, width, opacity }
  vectors = [], // { from: [x,y], to: [x,y], color, label, width }
  eigenLines = [], // { angle, color, label, dashed }
  points = [], // { x, y, color, fill, r, label, hollow }
  onPointDrag, // (idx, x, y) => void
  dragHandleIdx,
  origin = false,
  gridLines = true,
  title,
  axesLabels = ['g₁', 'g₂'],
}) => {
  const svgRef = useRef(null);
  const [dragging, setDragging] = useState(null);
  const x2px = (x) => size / 2 + (x / range) * (size / 2 - 12);
  const y2px = (y) => size / 2 - (y / range) * (size / 2 - 12);
  const px2x = (px) => ((px - size / 2) / (size / 2 - 12)) * range;
  const px2y = (py) => -((py - size / 2) / (size / 2 - 12)) * range;

  const handleMove = useCallback(
    (e) => {
      if (dragging === null || !svgRef.current) return;
      const r = svgRef.current.getBoundingClientRect();
      const px = e.clientX - r.left;
      const py = e.clientY - r.top;
      const x = Math.max(-range, Math.min(range, px2x(px)));
      const y = Math.max(-range, Math.min(range, px2y(py)));
      onPointDrag(dragging, x, y);
    },
    [dragging, range, onPointDrag, size]
  );

  useEffect(() => {
    if (dragging === null) return;
    const up = () => setDragging(null);
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', up);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', up);
    };
  }, [dragging, handleMove]);

  const pathFor = (data) =>
    data.length === 0
      ? ''
      : data
          .map((p, i) => `${i === 0 ? 'M' : 'L'} ${x2px(p[0]).toFixed(2)} ${y2px(p[1]).toFixed(2)}`)
          .join(' ');

  return (
    <svg
      ref={svgRef}
      width={size}
      height={size}
      style={{ background: C.panel, cursor: dragging !== null ? 'grabbing' : 'default', borderRadius: 2 }}
    >
      {title && (
        <text
          x={12}
          y={18}
          style={{ fontFamily: 'Fraunces, serif', fontSize: 13, fontStyle: 'italic', fill: C.ink }}
        >
          {title}
        </text>
      )}

      {/* grid */}
      {gridLines && (
        <>
          {[-2, -1, 1, 2].map((v) => (
            <g key={`g${v}`}>
              <line
                x1={x2px(v)}
                y1={6}
                x2={x2px(v)}
                y2={size - 6}
                stroke={C.border}
                strokeWidth={0.5}
              />
              <line
                x1={6}
                y1={y2px(v)}
                x2={size - 6}
                y2={y2px(v)}
                stroke={C.border}
                strokeWidth={0.5}
              />
            </g>
          ))}
        </>
      )}

      {/* axes */}
      <line x1={x2px(-range)} y1={size / 2} x2={x2px(range)} y2={size / 2} stroke={C.faint} strokeWidth={1} />
      <line x1={size / 2} y1={y2px(-range)} x2={size / 2} y2={y2px(range)} stroke={C.faint} strokeWidth={1} />

      {/* axis labels */}
      <text
        x={size - 8}
        y={size / 2 - 6}
        textAnchor="end"
        style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontSize: 11, fill: C.muted }}
      >
        {axesLabels[0]}
      </text>
      <text
        x={size / 2 + 6}
        y={14}
        style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontSize: 11, fill: C.muted }}
      >
        {axesLabels[1]}
      </text>

      {/* eigen lines */}
      {eigenLines.map((el, i) => {
        const c = Math.cos(el.angle), s = Math.sin(el.angle);
        const r = range * 0.92;
        return (
          <g key={`el${i}`}>
            <line
              x1={x2px(-r * c)}
              y1={y2px(-r * s)}
              x2={x2px(r * c)}
              y2={y2px(r * s)}
              stroke={el.color}
              strokeWidth={1}
              strokeDasharray={el.dashed === false ? 'none' : '4 3'}
              opacity={0.55}
            />
            {el.label && (
              <text
                x={x2px(r * c * 0.95)}
                y={y2px(r * s * 0.95) - 5}
                textAnchor="middle"
                style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontSize: 10, fill: el.color }}
              >
                {el.label}
              </text>
            )}
          </g>
        );
      })}

      {/* trajectories */}
      {trajectories.map((t, i) => (
        <path
          key={`tr${i}`}
          d={pathFor(t.data)}
          stroke={t.color}
          fill="none"
          strokeWidth={t.width || 2}
          strokeDasharray={t.dashed ? '5 4' : 'none'}
          opacity={t.opacity || 1}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      ))}

      {/* vectors */}
      {vectors.map((vec, i) => {
        const fx = x2px(vec.from[0]), fy = y2px(vec.from[1]);
        const tx = x2px(vec.to[0]), ty = y2px(vec.to[1]);
        const dx = tx - fx, dy = ty - fy;
        const len = Math.hypot(dx, dy);
        if (len < 2) return null;
        const ux = dx / len, uy = dy / len;
        const arrowL = 7;
        const arrowW = 4;
        const headX = tx, headY = ty;
        const leftX = headX - arrowL * ux - arrowW * (-uy);
        const leftY = headY - arrowL * uy - arrowW * ux;
        const rightX = headX - arrowL * ux + arrowW * (-uy);
        const rightY = headY - arrowL * uy + arrowW * ux;
        return (
          <g key={`v${i}`}>
            <line
              x1={fx}
              y1={fy}
              x2={tx - arrowL * 0.5 * ux}
              y2={ty - arrowL * 0.5 * uy}
              stroke={vec.color}
              strokeWidth={vec.width || 2}
              strokeLinecap="round"
            />
            <polygon
              points={`${headX},${headY} ${leftX},${leftY} ${rightX},${rightY}`}
              fill={vec.color}
            />
            {vec.label && (
              <text
                x={tx + 8 * ux}
                y={ty + 8 * uy + 4}
                style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontSize: 11, fill: vec.color }}
              >
                {vec.label}
              </text>
            )}
          </g>
        );
      })}

      {/* points */}
      {points.map((p, i) => (
        <g key={`p${i}`}>
          <circle
            cx={x2px(p.x)}
            cy={y2px(p.y)}
            r={p.r || 6}
            fill={p.hollow ? C.panel : p.fill || p.color}
            stroke={p.color}
            strokeWidth={2}
            style={{ cursor: onPointDrag && dragHandleIdx === i ? 'grab' : 'default' }}
            onMouseDown={(e) => {
              if (onPointDrag && dragHandleIdx === i) {
                e.preventDefault();
                setDragging(i);
              }
            }}
          />
          {p.label && (
            <text
              x={x2px(p.x) + 10}
              y={y2px(p.y) - 8}
              style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontSize: 11, fill: p.color }}
            >
              {p.label}
            </text>
          )}
        </g>
      ))}
    </svg>
  );
};

// ============================================================================
// CASE 1: scalar, constant rate
// ============================================================================

function Case1() {
  const [a, setA] = useState(0.7);
  const [x0, setX0] = useState(1.0);
  const [t, setT] = useState(2.0);

  const T = 8;
  const traj = useMemo(() => {
    const pts = [];
    const N = 200;
    for (let i = 0; i <= N; i++) {
      const ti = (i / N) * T;
      pts.push({ x: ti, y: x0 * Math.exp(-a * ti) });
    }
    return pts;
  }, [a, x0]);

  const xAtT = x0 * Math.exp(-a * t);
  const dotXAtT = -a * xAtT;
  const halfLife = Math.log(2) / a;
  const yMax = Math.max(Math.abs(x0), 0.1) * 1.1;
  const yMin = x0 < 0 ? -yMax : -0.1 * yMax;

  return (
    <div className="space-y-5">
      <Eq>
        <span>ẋ(t) = −a · x(t),</span>
        <span style={{ marginLeft: 12 }}>x(0) given</span>
        <span style={{ marginLeft: 12, color: C.muted }}>⟹</span>
        <span style={{ marginLeft: 12 }}>x(t) = x(0) e<sup>−at</sup></span>
      </Eq>

      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 lg:col-span-8" style={{ background: C.panel, padding: 16, border: `1px solid ${C.border}` }}>
          <LinePlot
            width={620}
            height={300}
            xRange={[0, T]}
            yRange={[yMin, yMax]}
            xLabel="t"
            yLabel="x(t)"
            xTicks={[0, 2, 4, 6, 8]}
            yTicks={[yMin, 0, yMax / 2, yMax]}
            series={[{ data: traj, color: C.burgundy, width: 2.5 }]}
            vlines={[
              { x: t, color: C.ink, label: `t = ${t.toFixed(2)}` },
              { x: halfLife, color: C.copper, dashed: true, label: 't₁/₂' },
            ]}
            hlines={[
              { y: 0, color: C.faint },
              { y: x0 / 2, color: C.copper, dashed: true, label: 'x(0)/2' },
            ]}
            markers={[
              { x: 0, y: x0, color: C.burgundy, r: 5, fill: C.panel, strokeWidth: 2 },
              { x: t, y: xAtT, color: C.ink, r: 5 },
            ]}
          />
        </div>
        <div className="col-span-12 lg:col-span-4 space-y-4" style={{ background: C.panel, padding: 16, border: `1px solid ${C.border}` }}>
          <Slider label="rate a" value={a} onChange={setA} min={0.05} max={3} step={0.01} />
          <Slider label="initial x(0)" value={x0} onChange={setX0} min={-2} max={2} step={0.01} />
          <Slider label="time t" value={t} onChange={setT} min={0} max={T} step={0.01} />
          <div className="pt-2 grid grid-cols-2 gap-3" style={{ borderTop: `1px solid ${C.border}` }}>
            <Stat label="x(t)" value={xAtT.toFixed(3)} color={C.ink} />
            <Stat label="ẋ(t)" value={dotXAtT.toFixed(3)} color={C.copper} />
            <Stat label="t₁/₂" value={halfLife.toFixed(2)} color={C.copper} />
            <Stat label="x(t)/x(0)" value={(xAtT / (x0 || 1)).toFixed(3)} color={C.muted} />
          </div>
        </div>
      </div>

      <div className="text-sm leading-relaxed" style={{ color: C.ink, maxWidth: 820 }}>
        <p style={{ fontFamily: 'Fraunces, serif', fontSize: '0.95rem' }}>
          The rate-of-change at every moment is proportional to the current value, with proportionality constant{' '}
          <em>−a</em>. The solution is exponential decay: each multiplication of <em>t</em> by{' '}
          <em>ln(2)/a</em> halves <em>x</em>, regardless of where you started. This is the
          one-dimensional version of gradient flow on a quadratic loss <em>L(w) = ½ a w²</em>{' '}
          — same ODE, same closed-form, and the rate <em>a</em> here is the eigenvalue of the
          Hessian/NTK in the 1D case.
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// CASE 2: scalar, time-varying rate
// ============================================================================

function Case2() {
  const [mode, setMode] = useState('ramp');
  const [a0, setA0] = useState(0.2);
  const [a1, setA1] = useState(1.5);
  const [tStar, setTStar] = useState(3.0);
  const [omega, setOmega] = useState(1.5);
  const [x0, setX0] = useState(1.0);
  const [t, setT] = useState(5.0);

  const T = 8;
  const N = 400;

  const aFn = useCallback(
    (tau) => {
      if (mode === 'ramp') {
        // smooth sigmoid step from a0 to a1 centered at tStar
        return a0 + (a1 - a0) / (1 + Math.exp(-(tau - tStar) * 2.5));
      } else {
        return Math.max(0.02, a0 + a1 * Math.sin(omega * tau) * 0.5);
      }
    },
    [mode, a0, a1, tStar, omega]
  );

  const { aData, intData, xData, naiveData } = useMemo(() => {
    const aArr = [];
    const intArr = [];
    const xArr = [];
    const naiveArr = [];
    const dt = T / N;
    let intA = 0;
    let prevA = aFn(0);
    aArr.push({ x: 0, y: prevA });
    intArr.push({ x: 0, y: 0 });
    xArr.push({ x: 0, y: x0 });
    naiveArr.push({ x: 0, y: x0 });
    // naive baseline uses a(0)
    const aBaseline = prevA;
    for (let i = 1; i <= N; i++) {
      const tau = i * dt;
      const av = aFn(tau);
      intA += (dt * (prevA + av)) / 2;
      aArr.push({ x: tau, y: av });
      intArr.push({ x: tau, y: intA });
      xArr.push({ x: tau, y: x0 * Math.exp(-intA) });
      naiveArr.push({ x: tau, y: x0 * Math.exp(-aBaseline * tau) });
      prevA = av;
    }
    return { aData: aArr, intData: intArr, xData: xArr, naiveData: naiveArr };
  }, [aFn, x0]);

  // values at current t
  const idx = Math.min(N, Math.round((t / T) * N));
  const aT = aData[idx].y;
  const intAT = intData[idx].y;
  const xT = xData[idx].y;
  const naiveT = naiveData[idx].y;

  const aMax = Math.max(...aData.map((d) => d.y)) * 1.15;
  const aMin = Math.min(0, ...aData.map((d) => d.y));
  const intMax = intData[N].y * 1.05;

  return (
    <div className="space-y-5">
      <Eq>
        <span>ẋ(t) = −a(t) · x(t)</span>
        <span style={{ marginLeft: 12, color: C.muted }}>⟹</span>
        <span style={{ marginLeft: 12 }}>x(t) = x(0) exp(−∫₀ᵗ a(τ) dτ)</span>
      </Eq>

      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 lg:col-span-8 space-y-4">
          <div style={{ background: C.panel, padding: 14, border: `1px solid ${C.border}` }}>
            <LinePlot
              width={620}
              height={170}
              xRange={[0, T]}
              yRange={[aMin, aMax]}
              xTicks={[0, 2, 4, 6, 8]}
              yTicks={[0, aMax / 2, aMax]}
              yLabel="a(τ)"
              title="rate function a(τ)"
              series={[{ data: aData, color: C.teal, width: 2.5 }]}
              vlines={[{ x: t, color: C.ink }]}
              markers={[{ x: t, y: aT, color: C.teal, r: 4 }]}
            />
          </div>
          <div style={{ background: C.panel, padding: 14, border: `1px solid ${C.border}` }}>
            <LinePlot
              width={620}
              height={170}
              xRange={[0, T]}
              yRange={[0, intMax]}
              xTicks={[0, 2, 4, 6, 8]}
              yTicks={[0, intMax / 2, intMax]}
              yLabel="∫a"
              title="integrated rate ∫₀ᵗ a(τ) dτ"
              series={[{ data: intData, color: C.copper, width: 2.5 }]}
              vlines={[{ x: t, color: C.ink }]}
              markers={[{ x: t, y: intAT, color: C.copper, r: 4 }]}
            />
          </div>
          <div style={{ background: C.panel, padding: 14, border: `1px solid ${C.border}` }}>
            <LinePlot
              width={620}
              height={210}
              xRange={[0, T]}
              yRange={[0, Math.max(Math.abs(x0), 0.1) * 1.1]}
              xTicks={[0, 2, 4, 6, 8]}
              yTicks={[0, Math.abs(x0) / 2, Math.abs(x0)]}
              yLabel="x(t)"
              xLabel="t"
              title="solution x(t) — true vs naive constant-rate baseline"
              series={[
                { data: naiveData, color: C.naive, width: 2, dashed: true },
                { data: xData, color: C.burgundy, width: 2.5 },
              ]}
              vlines={[{ x: t, color: C.ink }]}
              markers={[
                { x: t, y: xT, color: C.burgundy, r: 4 },
                { x: t, y: naiveT, color: C.naive, r: 4 },
              ]}
            />
          </div>
        </div>
        <div className="col-span-12 lg:col-span-4 space-y-4" style={{ background: C.panel, padding: 16, border: `1px solid ${C.border}` }}>
          <div>
            <div className="text-xs uppercase tracking-wider mb-2" style={{ color: C.muted }}>shape of a(t)</div>
            <Toggle
              options={[
                { value: 'ramp', label: 'ramp' },
                { value: 'osc', label: 'oscillate' },
              ]}
              value={mode}
              onChange={setMode}
            />
          </div>
          <Slider label="baseline a₀" value={a0} onChange={setA0} min={0.01} max={2} step={0.01} />
          {mode === 'ramp' ? (
            <>
              <Slider label="final a₁" value={a1} onChange={setA1} min={0.01} max={3} step={0.01} />
              <Slider label="ramp center t*" value={tStar} onChange={setTStar} min={0.5} max={7.5} step={0.05} />
            </>
          ) : (
            <>
              <Slider label="amplitude" value={a1} onChange={setA1} min={0} max={3} step={0.01} />
              <Slider label="frequency ω" value={omega} onChange={setOmega} min={0.2} max={6} step={0.05} />
            </>
          )}
          <Slider label="initial x(0)" value={x0} onChange={setX0} min={0.1} max={2} step={0.01} />
          <Slider label="time t" value={t} onChange={setT} min={0} max={T} step={0.01} />
          <div className="pt-2 space-y-2" style={{ borderTop: `1px solid ${C.border}` }}>
            <Stat label="a(t)" value={aT.toFixed(3)} color={C.teal} />
            <Stat label="∫₀ᵗ a" value={intAT.toFixed(3)} color={C.copper} />
            <Stat label="x(t) true" value={xT.toFixed(3)} color={C.burgundy} />
            <Stat label="x(t) naive" value={naiveT.toFixed(3)} color={C.naive} />
          </div>
        </div>
      </div>

      <div className="text-sm leading-relaxed" style={{ color: C.ink, maxWidth: 820 }}>
        <p style={{ fontFamily: 'Fraunces, serif', fontSize: '0.95rem' }}>
          With a time-varying rate, the exponent is no longer <em>−at</em>; it is the{' '}
          <em>integrated</em> rate over the elapsed interval. The naive constant-rate baseline
          (dashed) tracks decay only at the initial rate — when <em>a(t)</em> ramps up or
          oscillates, the true solution diverges from this baseline. Crucially, this works
          because scalars commute: only the running sum of rates matters, not the order in
          which they happen. That is what breaks in Case 4.
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// CASE 3: vector, constant matrix
// ============================================================================

function Case3() {
  const [lam1, setLam1] = useState(0.8);
  const [lam2, setLam2] = useState(0.2);
  const [theta, setTheta] = useState(Math.PI / 6);
  const [g0, setG0] = useState([1.6, 1.3]);
  const [t, setT] = useState(3.0);

  const T = 8;
  const N = 240;

  const trajectory = useMemo(() => {
    const pts = [];
    for (let i = 0; i <= N; i++) {
      const ti = (i / N) * T;
      pts.push(applyExpDecay(ti, lam1, lam2, theta, g0));
    }
    return pts;
  }, [lam1, lam2, theta, g0]);

  const gAtT = applyExpDecay(t, lam1, lam2, theta, g0);
  const trajUpToT = useMemo(() => {
    const cutoff = Math.round((t / T) * N);
    return trajectory.slice(0, cutoff + 1);
  }, [trajectory, t]);

  const norm = (v) => Math.hypot(v[0], v[1]);
  const normSeries = useMemo(
    () => trajectory.map((g, i) => ({ x: (i / N) * T, y: norm(g) })),
    [trajectory]
  );

  // modal decomposition: project g onto eigenvectors
  const v1 = [Math.cos(theta), Math.sin(theta)];
  const v2 = [-Math.sin(theta), Math.cos(theta)];
  const mode1Coord = g0[0] * v1[0] + g0[1] * v1[1];
  const mode2Coord = g0[0] * v2[0] + g0[1] * v2[1];
  const mode1Series = useMemo(() => {
    const pts = [];
    for (let i = 0; i <= N; i++) {
      const ti = (i / N) * T;
      pts.push({ x: ti, y: mode1Coord * Math.exp(-lam1 * ti) });
    }
    return pts;
  }, [lam1, mode1Coord]);
  const mode2Series = useMemo(() => {
    const pts = [];
    for (let i = 0; i <= N; i++) {
      const ti = (i / N) * T;
      pts.push({ x: ti, y: mode2Coord * Math.exp(-lam2 * ti) });
    }
    return pts;
  }, [lam2, mode2Coord]);

  const yMax = Math.max(Math.abs(mode1Coord), Math.abs(mode2Coord), 0.1) * 1.1;

  return (
    <div className="space-y-5">
      <Eq>
        <span>ġ(t) = −M · g(t),</span>
        <span style={{ marginLeft: 10 }}>M constant</span>
        <span style={{ marginLeft: 12, color: C.muted }}>⟹</span>
        <span style={{ marginLeft: 12 }}>g(t) = exp(−tM) g(0) = V · diag(e<sup>−λᵢt</sup>) · V⁻¹ g(0)</span>
      </Eq>

      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 lg:col-span-5" style={{ background: C.panel, padding: 16, border: `1px solid ${C.border}` }}>
          <PhasePlot
            size={420}
            range={2.5}
            trajectories={[
              { data: trajectory, color: C.faint, dashed: true, width: 1.5, opacity: 0.55 },
              { data: trajUpToT, color: C.ink, width: 2.2 },
            ]}
            eigenLines={[
              { angle: theta, color: C.burgundy, label: 'v₁' },
              { angle: theta + Math.PI / 2, color: C.teal, label: 'v₂' },
            ]}
            points={[
              { x: g0[0], y: g0[1], color: C.burgundy, hollow: true, r: 6, label: 'g(0)' },
              { x: gAtT[0], y: gAtT[1], color: C.ink, fill: C.ink, r: 6, label: 'g(t)' },
            ]}
            dragHandleIdx={0}
            onPointDrag={(idx, x, y) => setG0([x, y])}
            title="output-space trajectory"
            axesLabels={['g₁', 'g₂']}
          />
        </div>
        <div className="col-span-12 lg:col-span-7 space-y-3">
          <div style={{ background: C.panel, padding: 12, border: `1px solid ${C.border}` }}>
            <LinePlot
              width={520}
              height={170}
              xRange={[0, T]}
              yRange={[-yMax, yMax]}
              xTicks={[0, 2, 4, 6, 8]}
              yTicks={[-yMax, 0, yMax]}
              yLabel="mode coord."
              title="modal decomposition (in eigenbasis V)"
              series={[
                { data: mode1Series, color: C.burgundy, width: 2.2 },
                { data: mode2Series, color: C.teal, width: 2.2 },
              ]}
              vlines={[{ x: t, color: C.ink }]}
              hlines={[{ y: 0, color: C.faint }]}
              markers={[
                { x: t, y: mode1Coord * Math.exp(-lam1 * t), color: C.burgundy, r: 4 },
                { x: t, y: mode2Coord * Math.exp(-lam2 * t), color: C.teal, r: 4 },
              ]}
            />
          </div>
          <div style={{ background: C.panel, padding: 12, border: `1px solid ${C.border}` }}>
            <LinePlot
              width={520}
              height={170}
              xRange={[0, T]}
              yRange={[0, norm(g0) * 1.1]}
              xTicks={[0, 2, 4, 6, 8]}
              yTicks={[0, norm(g0) / 2, norm(g0)]}
              yLabel="‖g(t)‖"
              xLabel="t"
              title="norm of g(t)"
              series={[{ data: normSeries, color: C.copper, width: 2.2 }]}
              vlines={[{ x: t, color: C.ink }]}
              markers={[{ x: t, y: norm(gAtT), color: C.copper, r: 4 }]}
            />
          </div>
          <div className="grid grid-cols-2 gap-4" style={{ background: C.panel, padding: 16, border: `1px solid ${C.border}` }}>
            <Slider label="λ₁ (fast)" value={lam1} onChange={setLam1} min={0} max={2} step={0.01} />
            <Slider label="λ₂ (slow)" value={lam2} onChange={setLam2} min={0} max={2} step={0.01} />
            <Slider
              label="eigenbasis angle θ"
              value={theta}
              onChange={setTheta}
              min={0}
              max={Math.PI}
              step={0.01}
              fmt={(v) => `${((v * 180) / Math.PI).toFixed(0)}°`}
            />
            <Slider label="time t" value={t} onChange={setT} min={0} max={T} step={0.01} />
          </div>
        </div>
      </div>

      <div className="text-sm leading-relaxed" style={{ color: C.ink, maxWidth: 820 }}>
        <p style={{ fontFamily: 'Fraunces, serif', fontSize: '0.95rem' }}>
          Drag <em>g(0)</em>. With a constant matrix the eigenbasis is fixed in space, and{' '}
          <em>g(t)</em> decomposes cleanly: each component along an eigenvector decays
          independently at rate <em>λᵢ</em>. The trajectory bends toward the slow direction
          (teal <em>v₂</em>) because the fast component (burgundy <em>v₁</em>) dies away first.
          This is the lazy / frozen-NTK regime: eigenvectors don't move, only lengths along them
          shrink.
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// CASE 4: vector, time-varying matrix
// ============================================================================

function Case4() {
  const [submode, setSubmode] = useState('rotate');
  const [lam1, setLam1] = useState(0.9);
  const [lam2, setLam2] = useState(0.15);
  const [theta0, setTheta0] = useState(Math.PI / 6);
  const [dTheta, setDTheta] = useState(Math.PI / 2);
  const [lam2Final, setLam2Final] = useState(1.0);
  const [tWake, setTWake] = useState(3.5);
  const [g0, setG0] = useState([1.7, 1.3]);
  const [t, setT] = useState(5.0);

  const T = 8;
  const N = 240;

  // M(t) for the chosen sub-mode
  const getM = useCallback(
    (tau) => {
      if (submode === 'rotate') {
        const th = theta0 + (dTheta * tau) / T;
        return symMfromEig(lam1, lam2, th);
      } else {
        // awaken: theta fixed, lam2 ramps via sigmoid
        const l2 = lam2 + (lam2Final - lam2) / (1 + Math.exp(-(tau - tWake) * 2.5));
        return symMfromEig(lam1, l2, theta0);
      }
    },
    [submode, lam1, lam2, theta0, dTheta, lam2Final, tWake]
  );

  // True trajectory via RK4
  const trueTraj = useMemo(() => integrateODE(g0, getM, T, N), [g0, getM]);
  // Naive trajectory: g_naive(t) = exp(-A(t)) g(0), where A(t) = ∫_0^t M(τ) dτ
  const intMSeries = useMemo(() => cumulativeIntM(getM, T, N), [getM]);
  const naiveTraj = useMemo(() => {
    return intMSeries.map(({ t: ti, A }) => {
      // exp(-A) = matExpSym(-A)
      const negA = [[-A[0][0], -A[0][1]], [-A[1][0], -A[1][1]]];
      const E = matExpSym(negA);
      return { t: ti, g: matVec(E, g0) };
    });
  }, [intMSeries, g0]);

  const idx = Math.min(N, Math.round((t / T) * N));
  const gTrueAt = trueTraj[idx].g;
  const gNaiveAt = naiveTraj[idx].g;

  // Current M's eigen-decomp for display
  const currentTheta = submode === 'rotate' ? theta0 + (dTheta * t) / T : theta0;
  const currentLam2 = submode === 'rotate' ? lam2 : lam2 + (lam2Final - lam2) / (1 + Math.exp(-(t - tWake) * 2.5));

  const trueTrajData = useMemo(() => trueTraj.map((p) => p.g), [trueTraj]);
  const naiveTrajData = useMemo(() => naiveTraj.map((p) => p.g), [naiveTraj]);
  const trueUpToT = trueTrajData.slice(0, idx + 1);
  const naiveUpToT = naiveTrajData.slice(0, idx + 1);

  // side series: eigenvalue/angle over time
  const angleSeries = useMemo(() => {
    if (submode === 'rotate') {
      const pts = [];
      for (let i = 0; i <= N; i++) {
        const ti = (i / N) * T;
        pts.push({ x: ti, y: ((theta0 + (dTheta * ti) / T) * 180) / Math.PI });
      }
      return pts;
    }
    return [
      { x: 0, y: (theta0 * 180) / Math.PI },
      { x: T, y: (theta0 * 180) / Math.PI },
    ];
  }, [submode, theta0, dTheta]);
  const lam2Series = useMemo(() => {
    const pts = [];
    for (let i = 0; i <= N; i++) {
      const ti = (i / N) * T;
      const l2 = submode === 'rotate' ? lam2 : lam2 + (lam2Final - lam2) / (1 + Math.exp(-(ti - tWake) * 2.5));
      pts.push({ x: ti, y: l2 });
    }
    return pts;
  }, [submode, lam2, lam2Final, tWake]);

  // divergence between true and naive
  const divergence = useMemo(() => {
    return trueTraj.map((p, i) => ({
      x: p.t,
      y: Math.hypot(p.g[0] - naiveTraj[i].g[0], p.g[1] - naiveTraj[i].g[1]),
    }));
  }, [trueTraj, naiveTraj]);
  const maxDiv = Math.max(0.01, ...divergence.map((d) => d.y)) * 1.1;

  const angleRange = (() => {
    const vals = angleSeries.map((p) => p.y);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = Math.max(5, (hi - lo) * 0.1);
    return [Math.floor((lo - pad) / 10) * 10, Math.ceil((hi + pad) / 10) * 10];
  })();
  const lamMax = Math.max(lam1, lam2Final, lam2) * 1.15;

  return (
    <div className="space-y-5">
      <Eq>
        <span>ġ(t) = −M(t) · g(t)</span>
        <span style={{ marginLeft: 12, color: C.muted }}>—</span>
        <span style={{ marginLeft: 12 }}>g(t) = P<sub>g</sub>(t, 0) g(0)</span>
        <span style={{ marginLeft: 12, color: C.muted }}>where</span>
        <span style={{ marginLeft: 12 }}>∂<sub>t</sub>P<sub>g</sub> = −M(t) P<sub>g</sub>,</span>
        <span style={{ marginLeft: 8 }}>P<sub>g</sub>(0,0) = I</span>
      </Eq>

      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 lg:col-span-5" style={{ background: C.panel, padding: 16, border: `1px solid ${C.border}` }}>
          <PhasePlot
            size={420}
            range={2.5}
            trajectories={[
              { data: naiveTrajData, color: C.naive, dashed: true, width: 1.5, opacity: 0.7 },
              { data: trueTrajData, color: C.faint, dashed: true, width: 1.5, opacity: 0.55 },
              { data: naiveUpToT, color: C.naive, width: 2 },
              { data: trueUpToT, color: C.ink, width: 2.4 },
            ]}
            eigenLines={[
              { angle: currentTheta, color: C.burgundy, label: 'v₁(t)' },
              { angle: currentTheta + Math.PI / 2, color: C.teal, label: 'v₂(t)' },
            ]}
            points={[
              { x: g0[0], y: g0[1], color: C.burgundy, hollow: true, r: 6, label: 'g(0)' },
              { x: gTrueAt[0], y: gTrueAt[1], color: C.ink, fill: C.ink, r: 6, label: 'g(t) true' },
              { x: gNaiveAt[0], y: gNaiveAt[1], color: C.naive, fill: C.naive, r: 5, label: 'naive' },
            ]}
            dragHandleIdx={0}
            onPointDrag={(idx2, x, y) => setG0([x, y])}
            title="true vs naive trajectory"
            axesLabels={['g₁', 'g₂']}
          />
        </div>
        <div className="col-span-12 lg:col-span-7 space-y-3">
          <div style={{ background: C.panel, padding: 12, border: `1px solid ${C.border}` }}>
            <LinePlot
              width={520}
              height={150}
              xRange={[0, T]}
              yRange={angleRange}
              xTicks={[0, 2, 4, 6, 8]}
              yTicks={[angleRange[0], (angleRange[0] + angleRange[1]) / 2, angleRange[1]]}
              yLabel="θ(t) °"
              title="eigenbasis angle θ(t)"
              series={[{ data: angleSeries, color: C.burgundy, width: 2.2 }]}
              vlines={[{ x: t, color: C.ink }]}
            />
          </div>
          <div style={{ background: C.panel, padding: 12, border: `1px solid ${C.border}` }}>
            <LinePlot
              width={520}
              height={150}
              xRange={[0, T]}
              yRange={[0, lamMax]}
              xTicks={[0, 2, 4, 6, 8]}
              yTicks={[0, lam1, lamMax]}
              yLabel="λᵢ(t)"
              title="eigenvalues"
              series={[
                { data: [{ x: 0, y: lam1 }, { x: T, y: lam1 }], color: C.burgundy, width: 2.2 },
                { data: lam2Series, color: C.teal, width: 2.2 },
              ]}
              vlines={[{ x: t, color: C.ink }]}
            />
          </div>
          <div style={{ background: C.panel, padding: 12, border: `1px solid ${C.border}` }}>
            <LinePlot
              width={520}
              height={150}
              xRange={[0, T]}
              yRange={[0, maxDiv]}
              xTicks={[0, 2, 4, 6, 8]}
              yTicks={[0, maxDiv / 2, maxDiv]}
              yLabel="‖true − naive‖"
              xLabel="t"
              title="naive-guess error  ‖g_true(t) − exp(−∫M) g(0)‖"
              series={[{ data: divergence, color: C.copper, width: 2.2 }]}
              vlines={[{ x: t, color: C.ink }]}
              markers={[{ x: t, y: divergence[idx].y, color: C.copper, r: 4 }]}
            />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 lg:col-span-8 grid grid-cols-2 gap-4" style={{ background: C.panel, padding: 16, border: `1px solid ${C.border}` }}>
          <div className="col-span-2">
            <div className="text-xs uppercase tracking-wider mb-2" style={{ color: C.muted }}>shape of M(t)</div>
            <Toggle
              options={[
                { value: 'rotate', label: 'rotate (non-commuting)' },
                { value: 'awaken', label: 'awaken (commuting)' },
              ]}
              value={submode}
              onChange={setSubmode}
            />
          </div>
          <Slider label="λ₁" value={lam1} onChange={setLam1} min={0.05} max={2} step={0.01} />
          <Slider
            label={submode === 'awaken' ? 'λ₂ initial' : 'λ₂'}
            value={lam2}
            onChange={setLam2}
            min={0}
            max={2}
            step={0.01}
          />
          <Slider
            label="initial angle θ₀"
            value={theta0}
            onChange={setTheta0}
            min={0}
            max={Math.PI}
            step={0.01}
            fmt={(v) => `${((v * 180) / Math.PI).toFixed(0)}°`}
          />
          {submode === 'rotate' ? (
            <Slider
              label="rotation Δθ"
              value={dTheta}
              onChange={setDTheta}
              min={0}
              max={Math.PI}
              step={0.01}
              fmt={(v) => `${((v * 180) / Math.PI).toFixed(0)}°`}
            />
          ) : (
            <>
              <Slider label="λ₂ final" value={lam2Final} onChange={setLam2Final} min={0} max={2} step={0.01} />
              <Slider label="awaken time" value={tWake} onChange={setTWake} min={0.5} max={7} step={0.05} />
            </>
          )}
          <Slider label="time t" value={t} onChange={setT} min={0} max={T} step={0.01} />
        </div>
        <div className="col-span-12 lg:col-span-4 space-y-2" style={{ background: C.panel, padding: 16, border: `1px solid ${C.border}` }}>
          <Stat label="θ(t)" value={`${((currentTheta * 180) / Math.PI).toFixed(1)}°`} color={C.burgundy} />
          <Stat label="λ₂(t)" value={currentLam2.toFixed(3)} color={C.teal} />
          <Stat
            label="g(t) true"
            value={`(${gTrueAt[0].toFixed(2)}, ${gTrueAt[1].toFixed(2)})`}
            color={C.ink}
          />
          <Stat
            label="g(t) naive"
            value={`(${gNaiveAt[0].toFixed(2)}, ${gNaiveAt[1].toFixed(2)})`}
            color={C.naive}
          />
          <Stat
            label="naive error"
            value={divergence[idx].y.toFixed(3)}
            color={C.copper}
          />
        </div>
      </div>

      <div className="text-sm leading-relaxed" style={{ color: C.ink, maxWidth: 920 }}>
        <p style={{ fontFamily: 'Fraunces, serif', fontSize: '0.95rem' }}>
          <strong style={{ fontStyle: 'normal' }}>Rotate</strong> — the eigenvectors of <em>M(t)</em> sweep
          through the plane while eigenvalues stay fixed. <em>M(τ₁)</em> and <em>M(τ₂)</em> no longer
          commute, so the naive guess <em>exp(−∫₀ᵗ M(τ) dτ) g(0)</em> diverges from the true
          trajectory (copper error curve grows). The propagator <em>P<sub>g</sub>(t, 0)</em> from Eq. 7
          is the well-defined object that fixes this — it is, informally, the time-ordered product of
          infinitesimal exponentials <em>exp(−M(τ) dτ)</em> taken in trajectory order.
        </p>
        <p style={{ fontFamily: 'Fraunces, serif', fontSize: '0.95rem', marginTop: '0.5rem' }}>
          <strong style={{ fontStyle: 'normal' }}>Awaken</strong> — the eigenvalue <em>λ₂</em> ramps from
          near zero to a positive value, but the eigenbasis stays fixed. <em>M(τ₁)</em> and <em>M(τ₂)</em>{' '}
          share an eigenbasis ⟹ they commute ⟹ the naive guess actually works, and the error stays at zero
          to integration precision. This is the commuting sub-case of Case 4 — it lives between Case 3 and
          full feature learning.
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// Main
// ============================================================================

export default function App() {
  const [mode, setMode] = useState('case1');

  useEffect(() => {
    const id = 'fonts-link';
    if (document.getElementById(id)) return;
    const link = document.createElement('link');
    link.id = id;
    link.rel = 'stylesheet';
    link.href =
      'https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400;1,9..144,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap';
    document.head.appendChild(link);
  }, []);

  return (
    <div
      style={{
        background: C.bg,
        color: C.text,
        fontFamily: 'IBM Plex Sans, system-ui, sans-serif',
        minHeight: '100vh',
        padding: '32px 24px',
      }}
    >
      <div className="max-w-6xl mx-auto">
        <header className="mb-7">
          <div className="flex items-baseline justify-between flex-wrap gap-4">
            <div>
              <h1
                style={{
                  fontFamily: 'Fraunces, serif',
                  fontSize: '2.1rem',
                  fontWeight: 500,
                  letterSpacing: '-0.02em',
                  color: C.ink,
                }}
              >
                How do <em>ODEs</em> work?
              </h1>
              <p
                style={{
                  fontFamily: 'Fraunces, serif',
                  fontStyle: 'italic',
                  color: C.muted,
                  marginTop: 4,
                  fontSize: '1rem',
                }}
              >
                Four cases of linear ODEs, building up to the propagator — companion to Eqs. 1–7 of Litman & Guo.
              </p>
            </div>
            <div className="text-xs" style={{ color: C.muted, maxWidth: 280 }}>
              <span style={{ color: C.burgundy }}>v₁</span> fast direction ·{' '}
              <span style={{ color: C.teal }}>v₂</span> slow direction ·{' '}
              <span style={{ color: C.copper }}>copper</span> rate / error ·{' '}
              <span style={{ color: C.naive }}>grey</span> naive guess
            </div>
          </div>
        </header>

        <div className="flex flex-wrap gap-0 mb-6" style={{ borderBottom: `1px solid ${C.border}` }}>
          <Tab active={mode === 'case1'} onClick={() => setMode('case1')} sub="ẋ = −a·x">
            Case 1 — scalar, constant
          </Tab>
          <Tab active={mode === 'case2'} onClick={() => setMode('case2')} sub="ẋ = −a(t)·x">
            Case 2 — scalar, time-varying
          </Tab>
          <Tab active={mode === 'case3'} onClick={() => setMode('case3')} sub="ġ = −M·g">
            Case 3 — vector, constant
          </Tab>
          <Tab active={mode === 'case4'} onClick={() => setMode('case4')} sub="ġ = −M(t)·g">
            Case 4 — vector, time-varying
          </Tab>
        </div>

        {mode === 'case1' && <Case1 />}
        {mode === 'case2' && <Case2 />}
        {mode === 'case3' && <Case3 />}
        {mode === 'case4' && <Case4 />}

        <footer
          className="mt-12 pt-6 text-xs"
          style={{ borderTop: `1px solid ${C.border}`, color: C.muted, fontFamily: 'Fraunces, serif', fontStyle: 'italic' }}
        >
          Case 4 is the headline: in the rotate sub-mode, the eigenvectors of <em>M(t)</em> rotate, the matrices at
          different times stop commuting, and the naive exponential formula breaks. The propagator is what fixes
          this — and is precisely the object Litman & Guo need to integrate the output dynamics through feature
          learning.
        </footer>
      </div>
    </div>
  );
}
