import './style.css';
import { formatComplex, parseCoefficientList, type Complex } from '../core/complex';
import { leadingFixable } from '../core/enumeration';
import { nextBudget, previewDegree } from '../core/schedule';
import { fitView, panBy, pixelToWorld, worldToPixel, zoomAt, type View } from '../core/view';
import { getDevice } from '../gpu/device';
import { Engine, MAX_COEFFS, MAX_DEGREE, type DisplayParams } from '../gpu/engine';

const TARGET_MS = 12; // GPU time per frame; leaves headroom for 60 Hz
const MAX_ITER = 32;

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = el<HTMLCanvasElement>('view');
const handles = el<HTMLDivElement>('handles');
const degreeInput = el<HTMLInputElement>('degree');
const coeffInput = el<HTMLInputElement>('coeffs');
const colorSelect = el<HTMLSelectElement>('color');
const exposureInput = el<HTMLInputElement>('exposure');
const gammaInput = el<HTMLInputElement>('gamma');
const statusOut = el<HTMLOutputElement>('status');

function showError(message: string): void {
  const box = el<HTMLDivElement>('error');
  box.textContent = message;
  box.hidden = false;
}

const fmtTime = (s: number) => (s < 120 ? `${s.toFixed(0)} s` : s < 7200 ? `${(s / 60).toFixed(0)} min` : `${(s / 3600).toFixed(1)} h`);

async function start(): Promise<void> {
  const device = await getDevice();
  device.lost.then((info) => showError(`GPU device lost (${info.reason}): ${info.message}. Reload the page.`));
  const format = navigator.gpu.getPreferredCanvasFormat();
  const ctx = canvas.getContext('webgpu');
  if (!ctx) throw new Error('Could not create a WebGPU canvas context.');
  ctx.configure({ device, format, alphaMode: 'opaque' });
  const engine = new Engine(device, format);

  let coeffs: Complex[] = parseCoefficientList(coeffInput.value) ?? [{ re: 1, im: 0 }, { re: -1, im: 0 }];
  let degree = Number(degreeInput.value);
  let view: View | null = null;
  let interacting = false;
  let budget = 1 << 16;
  let inFlight = false;
  let displayDirty = true;
  let rootsPerSec = 0;
  let wheelTimer = 0;

  const display = (): DisplayParams => ({
    colorMode: colorSelect.value === '0' ? 0 : 1,
    exposure: Number(exposureInput.value),
    gamma: Number(gammaInput.value),
  });
  const dpr = () => canvas.width / canvas.clientWidth;
  const effectiveDegree = () => (interacting ? previewDegree(coeffs.length, degree, leadingFixable(coeffs), budget) : degree);
  const eventToPixel = (e: PointerEvent | WheelEvent) => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * dpr(), y: (e.clientY - r.top) * dpr() };
  };

  function restart(): void {
    if (!view) return;
    engine.setParams({ coeffs, degree: effectiveDegree(), view, maxIter: MAX_ITER });
    displayDirty = true;
    placeHandles();
  }

  function sizeCanvas(): void {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const maxDim = device.limits.maxTextureDimension2D;
    let w = Math.min(maxDim, Math.max(1, Math.round(canvas.clientWidth * ratio)));
    let h = Math.min(maxDim, Math.max(1, Math.round(canvas.clientHeight * ratio)));
    while (w * h * 16 > device.limits.maxStorageBufferBindingSize) {
      w = Math.floor(w * 0.9);
      h = Math.floor(h * 0.9);
    }
    canvas.width = w;
    canvas.height = h;
    engine.resize(w, h);
    view ??= fitView(w, h, 2);
    restart();
  }

  // ---- coefficient handles (DOM overlay, positioned in CSS pixels) ----
  function makeHandle(i: number): HTMLDivElement {
    const h = document.createElement('div');
    h.className = 'handle';
    h.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
      h.setPointerCapture(e.pointerId);
      interacting = true;
    });
    h.addEventListener('pointermove', (e) => {
      if (!h.hasPointerCapture(e.pointerId) || !view) return;
      const p = eventToPixel(e);
      const w = pixelToWorld(view, canvas.width, canvas.height, p.x, p.y);
      coeffs = coeffs.map((c, j) => (j === i ? w : c));
      coeffInput.value = coeffs.map(formatComplex).join(', ');
      restart();
    });
    const end = (e: PointerEvent) => {
      if (h.hasPointerCapture(e.pointerId)) h.releasePointerCapture(e.pointerId);
      interacting = false;
      restart();
    };
    h.addEventListener('pointerup', end);
    h.addEventListener('pointercancel', end);
    return h;
  }

  function placeHandles(): void {
    while (handles.children.length < coeffs.length) handles.appendChild(makeHandle(handles.children.length));
    while (handles.children.length > coeffs.length) handles.lastElementChild!.remove();
    if (!view) return;
    coeffs.forEach((c, i) => {
      const p = worldToPixel(view!, canvas.width, canvas.height, c.re, c.im);
      const h = handles.children[i] as HTMLDivElement;
      h.style.left = `${p.x / dpr()}px`;
      h.style.top = `${p.y / dpr()}px`;
      h.title = formatComplex(c);
    });
  }

  // ---- pan / zoom ----
  let panLast: { x: number; y: number } | null = null;
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    panLast = { x: e.clientX, y: e.clientY };
    interacting = true;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!panLast || !view) return;
    view = panBy(view, (e.clientX - panLast.x) * dpr(), (e.clientY - panLast.y) * dpr());
    panLast = { x: e.clientX, y: e.clientY };
    restart();
  });
  const endPan = () => {
    if (!panLast) return;
    panLast = null;
    interacting = false;
    restart();
  };
  canvas.addEventListener('pointerup', endPan);
  canvas.addEventListener('pointercancel', endPan);
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      if (!view) return;
      const p = eventToPixel(e);
      view = zoomAt(view, canvas.width, canvas.height, p.x, p.y, Math.exp(-e.deltaY * 0.0015));
      interacting = true;
      clearTimeout(wheelTimer);
      wheelTimer = window.setTimeout(() => {
        interacting = false;
        restart();
      }, 200);
      restart();
    },
    { passive: false },
  );
  const zoomCentre = (f: number) => {
    if (!view) return;
    view = zoomAt(view, canvas.width, canvas.height, canvas.width / 2, canvas.height / 2, f);
    restart();
  };
  el('zoomIn').addEventListener('click', () => zoomCentre(1.5));
  el('zoomOut').addEventListener('click', () => zoomCentre(1 / 1.5));
  el('reset').addEventListener('click', () => {
    view = fitView(canvas.width, canvas.height, 2);
    restart();
  });

  // ---- controls ----
  degreeInput.max = String(MAX_DEGREE);
  degreeInput.addEventListener('change', () => {
    degree = Math.min(MAX_DEGREE, Math.max(1, Math.round(Number(degreeInput.value)) || 1));
    degreeInput.value = String(degree);
    restart();
  });
  coeffInput.addEventListener('change', () => {
    const list = parseCoefficientList(coeffInput.value);
    const ok = list !== null && list.length <= MAX_COEFFS;
    coeffInput.classList.toggle('invalid', !ok);
    if (list && ok) {
      coeffs = list;
      restart();
    }
  });
  for (const input of [colorSelect, exposureInput, gammaInput]) input.addEventListener('input', () => (displayDirty = true));

  function updateStatus(): void {
    const { done, total } = engine.progress;
    const pct = total > 0n ? Number((done * 10000n) / total) / 100 : 100;
    const secondsLeft = (Number(total - done) * engine.degree) / Math.max(rootsPerSec, 1);
    statusOut.textContent =
      `degree ${engine.degree}${engine.degree < degree ? ' (preview)' : ''} · ${coeffs.length} coefficients\n` +
      `${pct.toFixed(2)} % of ${total.toLocaleString()} polynomials\n` +
      `${(rootsPerSec / 1e6).toFixed(1)} M roots/s` +
      (done < total ? ` · ~${fmtTime(secondsLeft)} left` : ' · done');
  }

  // ---- frame loop: at most one frame of GPU work in flight, so input never queues behind compute ----
  function frame(): void {
    requestAnimationFrame(frame);
    if (inFlight || !view) return;
    const enc = device.createCommandEncoder();
    const count = engine.encodeCompute(enc, budget);
    if (count === 0 && !displayDirty) return; // finished and nothing changed
    engine.encodeDisplay(enc, ctx!.getCurrentTexture().createView(), display());
    displayDirty = false;
    const deg = engine.degree;
    const t0 = performance.now();
    inFlight = true;
    device.queue.submit([enc.finish()]);
    device.queue.onSubmittedWorkDone().then(() => {
      inFlight = false;
      const ms = performance.now() - t0;
      if (count > 0) {
        budget = nextBudget(budget, ms, TARGET_MS);
        rootsPerSec = (count * deg) / (ms / 1000);
      }
      updateStatus();
    });
  }

  new ResizeObserver(() => sizeCanvas()).observe(canvas);
  requestAnimationFrame(frame);
}

start().catch((err: unknown) => showError(err instanceof Error ? err.message : String(err)));
