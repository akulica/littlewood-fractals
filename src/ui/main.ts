import './style.css';
import { frameTime, framesFor, interpolateAt, keyframeTime, type Easing, type Keyframe, type PathOptions } from '../core/animation';
import { formatComplex, parseCoefficientList, type Complex } from '../core/complex';
import { exportSizeError, exportView } from '../core/export';
import { encodePng } from '../core/png';
import { nextBudget, type Kernel } from '../core/schedule';
import { fitView, panBy, pixelToWorld, worldToPixel, zoomAt, type View } from '../core/view';
import { renderAnimation, type DraftQuality } from '../gpu/animation';
import { getDevice } from '../gpu/device';
import { Engine, MAX_COEFFS, MAX_DEGREE, renderImage, type Calibration, type DisplayParams } from '../gpu/engine';
import type { AnimationWriter, OutputKind } from './animationOutput';

const TARGET_MS = 12; // GPU time per frame while interacting; leaves headroom for 60 Hz
const IDLE_TARGET_MS = 40; // GPU time per back-to-back chunk while refining (well under the ~2 s Windows TDR)
const MAX_ITER = 32;

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = el<HTMLCanvasElement>('view');
const handles = el<HTMLDivElement>('handles');
const degreeInput = el<HTMLInputElement>('degree');
const coeffInput = el<HTMLInputElement>('coeffs');
const colorSelect = el<HTMLSelectElement>('color');
const kernelSelect = el<HTMLSelectElement>('kernel');
const exposureInput = el<HTMLInputElement>('exposure');
const gammaInput = el<HTMLInputElement>('gamma');
const statusOut = el<HTMLOutputElement>('status');
const exportWInput = el<HTMLInputElement>('exportW');
const exportHInput = el<HTMLInputElement>('exportH');
const saveButton = el<HTMLButtonElement>('savePng');
const exportStatus = el<HTMLOutputElement>('exportStatus');
const keyPaths = document.getElementById('keyPaths') as unknown as SVGSVGElement;
const keyCountOut = el<HTMLSpanElement>('keyCount');
const keyList = el<HTMLOListElement>('keyList');
const addKeyButton = el<HTMLButtonElement>('addKey');
const clearKeysButton = el<HTMLButtonElement>('clearKeys');
const timeline = el<HTMLInputElement>('timeline');
const playButton = el<HTMLButtonElement>('play');
const loopInput = el<HTMLInputElement>('loop');
const framesInput = el<HTMLInputElement>('frames');
const fpsInput = el<HTMLInputElement>('fps');
const easingSelect = el<HTMLSelectElement>('easing');
const brightnessSelect = el<HTMLSelectElement>('brightness');
const qualitySelect = el<HTMLSelectElement>('quality');
const outputSelect = el<HTMLSelectElement>('output');
const renderAnimButton = el<HTMLButtonElement>('renderAnim');
const cancelAnimButton = el<HTMLButtonElement>('cancelAnim');
const animStatus = el<HTMLOutputElement>('animStatus');
const SVG_NS = 'http://www.w3.org/2000/svg';
// The MP4 muxer is most of the bundle: a separate chunk, fetched in the background right away so the first
// picture is not delayed and the module is ready long before anyone clicks Render.
const outputModule = import('./animationOutput');

function showError(message: string): void {
  const box = el<HTMLDivElement>('error');
  box.textContent = message;
  box.hidden = false;
}

// 3.1, 0.25, 0.0012: enough digits to see a deep-degree sample grow.
const fmtPct = (p: number) => (p >= 0.1 || p === 0 ? p.toFixed(1) : p.toPrecision(2));
const sameView = (a: View, b: View) => a.cx === b.cx && a.cy === b.cy && a.pxPerUnit === b.pxPerUnit;
const sameCoeffs = (a: Complex[], b: Complex[]) => a.length === b.length && a.every((c, i) => c.re === b[i].re && c.im === b[i].im);

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
  let budget = 1 << 16; // interactive: polynomials per animation frame (TARGET_MS)
  let idleBudget = 1 << 16; // idle: polynomials per back-to-back chunk (IDLE_TARGET_MS)
  let inFlight = false;
  let pumping = false;
  let exporting = false; // export owns the GPU: the live view pauses its compute meanwhile
  let histChanged = false;
  let displayDirty = true;
  let rootsPerSec = 0;
  let wheelTimer = 0;
  let applied: { coeffs: Complex[]; degree: number; view: View } | null = null; // what the engine accumulates
  let keys: Keyframe[] = []; // animation keyframes, all of length coeffs.length
  let playing = false; // ▶ Preview: the timeline drives the coefficients; computes like an interaction meanwhile
  let animAbort: AbortController | null = null;

  const display = (): DisplayParams => ({
    colorMode: colorSelect.value === '0' ? 0 : 1,
    exposure: Number(exposureInput.value),
    gamma: Number(gammaInput.value),
  });
  const dpr = () => canvas.width / canvas.clientWidth;
  const eventToPixel = (e: PointerEvent | WheelEvent) => {
    const r = canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * dpr(), y: (e.clientY - r.top) * dpr() };
  };

  // Accumulation restarts only when the picture itself changed. The engine samples polynomials in golden order, so
  // whatever is on screen when a drag ends is already an even sample of the final picture: keep refining it.
  function restart(force = false): void {
    if (!view) return;
    const same = applied !== null && applied.degree === degree && sameView(applied.view, view) && sameCoeffs(applied.coeffs, coeffs);
    if (force || !same) {
      engine.setParams({ coeffs, degree, view, maxIter: MAX_ITER });
      applied = { coeffs, degree, view };
    }
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
      interacting = false; // pump() continues the same sample to the full set
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
    drawKeyPaths();
  }

  // Ghost rings where each keyframe puts every coefficient, and the path each coefficient travels (CSS pixels, like the handles).
  function drawKeyPaths(): void {
    keyPaths.replaceChildren();
    if (!view || keys.length === 0) return;
    const at = (c: Complex) => {
      const p = worldToPixel(view!, canvas.width, canvas.height, c.re, c.im);
      return { x: p.x / dpr(), y: p.y / dpr() };
    };
    const path = loopInput.checked && keys.length > 1 ? [...keys, keys[0]] : keys;
    for (let j = 0; j < keys[0].length; j++) {
      const line = document.createElementNS(SVG_NS, 'polyline');
      line.setAttribute('class', 'path');
      line.setAttribute('points', path.map((key) => { const p = at(key[j]); return `${p.x},${p.y}`; }).join(' '));
      keyPaths.appendChild(line);
      for (const key of keys) {
        const p = at(key[j]);
        const ring = document.createElementNS(SVG_NS, 'circle');
        ring.setAttribute('class', 'ghost');
        ring.setAttribute('cx', String(p.x));
        ring.setAttribute('cy', String(p.y));
        ring.setAttribute('r', '6');
        keyPaths.appendChild(ring);
      }
    }
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
  };
  canvas.addEventListener('pointerup', endPan);
  canvas.addEventListener('pointercancel', endPan);
  // On window, not the canvas: handles sit above the canvas and would otherwise swallow wheel events.
  window.addEventListener(
    'wheel',
    (e) => {
      if (e.target instanceof Element && e.target.closest('#panel')) return; // let the panel scroll normally
      e.preventDefault();
      if (!view) return;
      const p = eventToPixel(e);
      view = zoomAt(view, canvas.width, canvas.height, p.x, p.y, Math.exp(-e.deltaY * 0.0015));
      interacting = true;
      clearTimeout(wheelTimer);
      wheelTimer = window.setTimeout(() => (interacting = false), 200);
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
      if (keys.length > 0 && list.length !== keys[0].length) {
        // Keyframes interpolate coefficient by coefficient: they only make sense for one coefficient count.
        keys = [];
        stopPlay();
        updateKeys();
        setAnimStatus(`keyframes cleared: the number of coefficients changed to ${list.length}`);
      }
      coeffs = list;
      restart();
    }
  });
  for (const input of [colorSelect, exposureInput, gammaInput]) input.addEventListener('input', () => (displayDirty = true));
  kernelSelect.addEventListener('change', () => {
    engine.kernel = kernelSelect.value === 'auto' ? calibration.kernel : (kernelSelect.value as Kernel);
    restart(true); // re-render from scratch so the status line times the chosen kernel
  });

  function updateStatus(): void {
    const { done, total } = engine.progress;
    const pct = total > 0n ? Number((done * 10000n) / total) / 100 : 100;
    const secondsLeft = (Number(total - done) * engine.degree) / Math.max(rootsPerSec, 1);
    const sampled = done < total ? ` · preview · ${fmtPct((Number(done) / Number(total)) * 100)} % sampled` : '';
    const kernelNote = kernelSelect.value === 'auto' ? ` (auto: coop/perPoly = ${(calibration.coopMs / calibration.perPolyMs).toFixed(2)})` : '';
    statusOut.textContent =
      `degree ${engine.degree}${sampled} · ${coeffs.length} coefficients\n` +
      `kernel ${engine.kernel}${kernelNote}\n` +
      `${pct.toFixed(2)} % of ${total.toLocaleString()} polynomials\n` +
      `${(rootsPerSec / 1e6).toFixed(1)} M roots/s` +
      (done < total ? ` · ~${fmtTime(secondsLeft)} left` : ' · done');
  }

  // ---- interactive: compute + display in one submit per animation frame, at most one frame in flight,
  // so input never queues behind compute. Each frame adds the next golden-order slice at full degree, sized by
  // the adaptive budget, so the preview is a growing even sample of the final picture. Idle: pump() keeps the
  // GPU busy with back-to-back chunks and frame() only redraws (measured: tying idle compute to rAF left the GPU ~75 % idle at degree 24). ----
  async function pump(): Promise<void> {
    if (pumping) return;
    pumping = true;
    try {
      while (!interacting && !playing && !exporting) {
        const enc = device.createCommandEncoder();
        const count = engine.encodeCompute(enc, idleBudget);
        if (count === 0) break;
        const deg = engine.degree;
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        histChanged = true;
        await device.queue.onSubmittedWorkDone();
        const ms = performance.now() - t0;
        idleBudget = nextBudget(idleBudget, ms, IDLE_TARGET_MS);
        rootsPerSec = (count * deg) / (ms / 1000);
      }
    } finally {
      pumping = false;
    }
  }

  function frame(): void {
    requestAnimationFrame(frame);
    if (inFlight || !view) return;
    const enc = device.createCommandEncoder();
    let count = 0;
    if (exporting) {
      // live compute paused; still redraw if display settings change
    } else if (interacting || playing) count = engine.encodeCompute(enc, budget);
    else if (!pumping && engine.progress.done < engine.progress.total) void pump();
    if (count === 0 && !displayDirty && !histChanged) return; // nothing new to show
    engine.encodeDisplay(enc, ctx!.getCurrentTexture().createView(), display());
    displayDirty = false;
    histChanged = false;
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

  // ---- PNG export: re-render at the requested size off-screen (a private histogram), never upscale ----
  function setExportStatus(text: string, isError = false): void {
    exportStatus.textContent = text;
    exportStatus.classList.toggle('error', isError);
  }
  saveButton.addEventListener('click', async () => {
    if (!view || exporting) return;
    const width = Number(exportWInput.value);
    const height = Number(exportHInput.value);
    const problem = exportSizeError(width, height, device.limits);
    if (problem) return setExportStatus(problem, true);
    setExportStatus(`exporting ${width}×${height}: starting…`);
    exporting = true;
    saveButton.disabled = true;
    const t0 = performance.now();
    try {
      const outView = exportView(view, canvas.width, canvas.height, width, height);
      const density = outView.pxPerUnit / view.pxPerUnit; // export pixels per screen pixel, per axis
      const rgba = await renderImage(
        device,
        {
          kernel: engine.kernel,
          params: { coeffs, degree, view: outView, maxIter: MAX_ITER },
          display: { ...display(), countScale: density * density }, // same brightness as on screen
          width,
          height,
        },
        (done, total) => setExportStatus(`exporting ${width}×${height}: ${(Number((done * 1000n) / total) / 10).toFixed(1)} %`),
      );
      setExportStatus(`encoding PNG…`);
      const blob = await encodePng(rgba, width, height);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `littlewood-d${degree}-k${coeffs.length}-${width}x${height}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
      setExportStatus(`saved ${a.download} (${(blob.size / 2 ** 20).toFixed(1)} MB, ${((performance.now() - t0) / 1000).toFixed(1)} s)`);
    } catch (err) {
      setExportStatus(`export failed: ${err instanceof Error ? err.message : String(err)}`, true);
    } finally {
      exporting = false;
      saveButton.disabled = false;
      displayDirty = true;
    }
  });
  exportWInput.max = exportHInput.max = String(device.limits.maxTextureDimension2D);

  // ---- animation: keyframes -> interpolated coefficient sets -> one full render per frame -> MP4 / PNG frames ----
  const pathOpts = (): PathOptions => ({ easing: easingSelect.value as Easing, loop: loopInput.checked });
  const frameCount = () => Math.min(100_000, Math.max(2, Math.round(Number(framesInput.value)) || 2));
  const fps = () => Math.min(120, Math.max(1, Math.round(Number(fpsInput.value)) || 30));
  function setAnimStatus(text: string, isError = false): void {
    animStatus.textContent = text;
    animStatus.classList.toggle('error', isError);
  }
  function showCoeffs(list: Complex[]): void {
    coeffs = list;
    coeffInput.value = coeffs.map(formatComplex).join(', ');
    coeffInput.classList.remove('invalid');
    restart();
  }
  function updateKeys(): void {
    keyCountOut.textContent = keys.length > 0 ? `(${keys.length} keyframe${keys.length === 1 ? '' : 's'})` : '';
    keyList.replaceChildren(
      ...keys.map((key, i) => {
        const li = document.createElement('li');
        const label = document.createElement('span');
        label.textContent = key.map(formatComplex).join(', ');
        li.title = 'Show this keyframe';
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.textContent = '×';
        remove.title = 'Remove keyframe';
        remove.addEventListener('click', (e) => {
          e.stopPropagation();
          keys.splice(i, 1);
          if (keys.length < 2) stopPlay();
          updateKeys();
        });
        li.addEventListener('click', () => {
          stopPlay();
          if (keys.length >= 2) timeline.value = String(keyframeTime(i, keys.length, pathOpts()));
          showCoeffs(key);
        });
        li.append(label, remove);
        return li;
      }),
    );
    const usable = keys.length >= 2 && !exporting;
    timeline.disabled = !usable;
    playButton.disabled = !usable;
    drawKeyPaths();
  }

  addKeyButton.addEventListener('click', () => {
    keys.push(coeffs.slice());
    updateKeys();
    setAnimStatus(keys.length < 2 ? 'move the coefficients, then add the next keyframe' : '');
  });
  clearKeysButton.addEventListener('click', () => {
    keys = [];
    stopPlay();
    updateKeys();
  });
  loopInput.addEventListener('change', () => drawKeyPaths());

  // Scrubbing is an interaction: the live view renders the full-degree golden sample of each position.
  timeline.addEventListener('input', () => {
    if (keys.length < 2) return;
    stopPlay();
    interacting = true;
    showCoeffs(interpolateAt(keys, Number(timeline.value), pathOpts()));
  });
  timeline.addEventListener('change', () => (interacting = false));

  // ▶ Preview steps through the same frame positions the render will use, at the chosen fps, looping.
  let playStart = 0;
  let playFrame = -1;
  function playTick(now: number): void {
    if (!playing) return;
    const n = frameCount();
    const i = Math.floor(((now - playStart) / 1000) * fps()) % n;
    if (i !== playFrame) {
      playFrame = i;
      const t = frameTime(i, n, loopInput.checked);
      timeline.value = String(t);
      showCoeffs(interpolateAt(keys, t, pathOpts()));
    }
    requestAnimationFrame(playTick);
  }
  function stopPlay(): void {
    if (!playing) return;
    playing = false;
    playButton.textContent = '▶ Preview';
  }
  playButton.addEventListener('click', () => {
    if (playing) return stopPlay();
    if (keys.length < 2) return;
    playing = true;
    playStart = performance.now();
    playFrame = -1;
    playButton.textContent = '■ Stop';
    requestAnimationFrame(playTick);
  });

  void outputModule.then(({ canWriteFolders }) => {
    if (!canWriteFolders()) for (const option of [...outputSelect.options]) if (option.value !== 'mp4') option.remove();
  });

  renderAnimButton.addEventListener('click', async () => {
    if (!view || exporting) return;
    if (keys.length < 2) return setAnimStatus('add at least two keyframes first', true);
    const width = Number(exportWInput.value);
    const height = Number(exportHInput.value);
    const problem = exportSizeError(width, height, device.limits);
    if (problem) return setAnimStatus(problem, true);
    const kind = outputSelect.value as OutputKind;
    const n = frameCount();
    const rate = fps();
    const { mp4Error, openAnimationWriter } = await outputModule;
    if (kind !== 'png') {
      const mp4Problem = await mp4Error(width, height, rate);
      if (mp4Problem) return setAnimStatus(mp4Problem, true);
    }
    let writer: AnimationWriter;
    try {
      // First await after the encoder check: the file/folder picker needs this click's user activation.
      writer = await openAnimationWriter({ kind, width, height, fps: rate, name: `littlewood-d${degree}-k${coeffs.length}-${width}x${height}-${n}f` });
    } catch (err) {
      const dismissed = err instanceof DOMException && err.name === 'AbortError';
      return setAnimStatus(dismissed ? 'cancelled' : `could not open the output: ${err instanceof Error ? err.message : String(err)}`, !dismissed);
    }
    stopPlay();
    exporting = true;
    const ctl = new AbortController();
    animAbort = ctl;
    renderAnimButton.disabled = saveButton.disabled = true;
    cancelAnimButton.disabled = false;
    updateKeys();
    setAnimStatus(`rendering ${n} frames at ${width}×${height}…`);
    const t0 = performance.now();
    try {
      const outView = exportView(view, canvas.width, canvas.height, width, height);
      const density = outView.pxPerUnit / view.pxPerUnit; // export pixels per screen pixel, per axis
      await renderAnimation(
        device,
        {
          coeffFrames: framesFor(keys, n, pathOpts()),
          degree,
          view: outView,
          width,
          height,
          display: { ...display(), countScale: density * density }, // same brightness as on screen
          brightness: brightnessSelect.value === 'perFrame' ? 'perFrame' : 'locked',
          quality: Number(qualitySelect.value) as DraftQuality,
          maxIter: MAX_ITER,
        },
        (rgba, i) => writer.frame(rgba, i),
        (p) => {
          // Compute dominates while a batch runs, encoding while its frames go out: average the two fractions.
          const s = (performance.now() - t0) / 1000;
          const f = (p.polysDone / p.polysTotal + p.frame / p.frames) / 2;
          const perFrame = p.frame > 0 ? ` · ${((s * 1000) / p.frame).toFixed(0)} ms/frame` : '';
          const left = f > 0.01 ? ` · ~${fmtTime((s * (1 - f)) / f)} left` : '';
          setAnimStatus(`frame ${p.frame}/${p.frames} · ${(f * 100).toFixed(1)} %${perFrame}${left}`);
        },
        ctl.signal,
      );
      setAnimStatus('finishing…');
      const summary = await writer.finish();
      const s = (performance.now() - t0) / 1000;
      setAnimStatus(`${summary}\n${n} frames in ${fmtTime(s)} (${((s * 1000) / n).toFixed(0)} ms/frame)`);
    } catch (err) {
      await writer.cancel();
      if (ctl.signal.aborted) setAnimStatus('render cancelled');
      else setAnimStatus(`render failed: ${err instanceof Error ? err.message : String(err)}`, true);
    } finally {
      exporting = false;
      animAbort = null;
      renderAnimButton.disabled = saveButton.disabled = false;
      cancelAnimButton.disabled = true;
      updateKeys();
      displayDirty = true;
    }
  });
  cancelAnimButton.addEventListener('click', () => animAbort?.abort());

  // Pick the faster splat kernel for this GPU before the first frame (~0.1-0.5 s; the canvas stays black meanwhile).
  sizeCanvas();
  statusOut.textContent = 'measuring kernels on this GPU…';
  kernelSelect.disabled = true; // its listener reads `calibration`, which does not exist until this resolves
  const calibration: Calibration = await engine.calibrate();
  kernelSelect.disabled = false;
  restart(true); // calibrate() left its own polynomials loaded
  new ResizeObserver(() => sizeCanvas()).observe(canvas);
  requestAnimationFrame(frame);
}

start().catch((err: unknown) => showError(err instanceof Error ? err.message : String(err)));
