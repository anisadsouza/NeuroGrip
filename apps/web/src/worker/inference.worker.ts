/// <reference lib="webworker" />

/**
 * The inference worker.
 *
 * Owns the whole real-time path: ring buffer, windowing, signal-quality gate,
 * feature extraction, ONNX inference, and evidence accumulation. It emits one
 * small decision per 20 ms hop.
 *
 * All of it runs off the UI thread, so a slow frame in React can never delay a
 * decision, and a slow decision can never drop a frame. That separation is why
 * the latency figure this project treats as a safety property is measurable at
 * all: nothing else shares the thread it runs on.
 */

import * as ort from 'onnxruntime-web';
import {
  DEFAULT_FEATURE_CONFIG,
  DEFAULT_SIGNAL_QUALITY_CONFIG,
  DEFAULT_WINDOW_CONFIG,
  EvidenceAccumulator,
  MultiChannelRingBuffer,
  checkSignalQuality,
  featureVector,
  hopSamples,
  windowSamples,
} from '@neurogrip/core';
import {
  DISPLAY_POINTS,
  type DecisionResponse,
  type WorkerRequest,
  type WorkerResponse,
} from './protocol.js';

declare const self: DedicatedWorkerGlobalScope;

interface Runtime {
  session: ort.InferenceSession;
  inputName: string;
  ring: MultiChannelRingBuffer;
  accumulator: EvidenceAccumulator;
  nChannels: number;
  windowWidth: number;
  hop: number;
  gestures: readonly string[];
  /** Samples pushed since the last decision, so hops land on exact boundaries. */
  pending: number;
}

let runtime: Runtime | null = null;

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(message, transfer);
}

function fail(error: unknown): void {
  post({
    type: 'error',
    message: error instanceof Error ? error.message : String(error),
  });
}

async function init(
  modelUrl: string,
  nChannels: number,
  samplingRateHz: number,
  risks: readonly number[],
): Promise<void> {
  // Multi-threaded WASM needs SharedArrayBuffer, which needs cross-origin
  // isolation. Where that is unavailable the runtime still works, single
  // threaded and roughly twice as slow, so report which one we got rather than
  // quietly publishing an optimistic latency number.
  const threaded = typeof SharedArrayBuffer !== 'undefined';
  ort.env.wasm.numThreads = threaded
    ? Math.min(4, navigator.hardwareConcurrency || 1)
    : 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.wasmPaths = `${self.location.origin}/ort/`;

  const metadataUrl = modelUrl.replace(/\.onnx$/, '.json');
  const metadata = await fetch(metadataUrl).then((r) => r.json());

  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });

  const windowConfig = { ...DEFAULT_WINDOW_CONFIG, samplingRateHz };
  const windowWidth = windowSamples(windowConfig);

  runtime = {
    session,
    inputName: session.inputNames[0]!,
    // Two windows of capacity, so a burst of samples cannot overrun the
    // oldest data the current window still needs.
    ring: new MultiChannelRingBuffer(nChannels, windowWidth * 2),
    accumulator: new EvidenceAccumulator(risks),
    nChannels,
    windowWidth,
    hop: hopSamples(windowConfig),
    gestures: metadata.gestures as string[],
    pending: 0,
  };

  post({
    type: 'ready',
    gestures: runtime.gestures,
    nFeatures: metadata.n_features,
    backend: 'wasm',
    threaded,
    featureSpecVersion: metadata.feature_spec_version,
  });
}

/**
 * Decimate a window for drawing, preserving the envelope.
 *
 * Min/max decimation, not peak-picking. Taking the largest-magnitude sample in
 * each bucket loses the sign alternation that makes a waveform look like a
 * waveform: consecutive buckets can both land on positive peaks and the trace
 * drifts off the baseline into smooth undulation. Emitting the minimum and the
 * maximum of each bucket in order keeps both extremes, so the drawn envelope
 * matches what an oscilloscope would show.
 */
function decimate(channels: Float64Array[], points: number): Float32Array {
  const out = new Float32Array(channels.length * points);
  const width = channels[0]!.length;
  const buckets = Math.max(1, Math.floor(points / 2));
  const stride = width / buckets;

  for (let c = 0; c < channels.length; c++) {
    const source = channels[c]!;
    const base = c * points;
    for (let b = 0; b < buckets; b++) {
      const start = Math.floor(b * stride);
      const end = Math.max(start + 1, Math.min(width, Math.floor((b + 1) * stride)));

      let low = source[start]!;
      let high = source[start]!;
      for (let i = start + 1; i < end; i++) {
        const value = source[i]!;
        if (value < low) low = value;
        if (value > high) high = value;
      }

      // Alternate the order bucket to bucket so the polyline zig-zags through
      // the envelope rather than doubling back on itself every other point.
      const first = b % 2 === 0 ? low : high;
      const second = b % 2 === 0 ? high : low;
      out[base + b * 2] = first;
      if (b * 2 + 1 < points) out[base + b * 2 + 1] = second;
    }
  }
  return out;
}

async function decodeWindow(active: Runtime): Promise<void> {
  const window = active.ring.readLatestWindow(active.windowWidth);
  if (!window) return;

  const quality = checkSignalQuality(window, DEFAULT_SIGNAL_QUALITY_CONFIG);
  if (!quality.ok) {
    // A rejected window is not a prediction of "rest" -- it is a refusal to
    // predict. Letting it reach the accumulator would let a detached electrode
    // accrue evidence for whatever class its noise happens to resemble.
    post({
      type: 'rejected',
      reason: quality.reason ?? 'signal quality check failed',
      failedChannel: quality.failedChannel ?? null,
    });
    return;
  }

  const featureStart = performance.now();
  const features = featureVector(window, {
    ...DEFAULT_FEATURE_CONFIG,
    samplingRateHz: DEFAULT_WINDOW_CONFIG.samplingRateHz,
  });
  const featureEnd = performance.now();

  const tensor = new ort.Tensor('float32', features, [1, features.length]);
  const output = await active.session.run({ [active.inputName]: tensor });
  const inferenceEnd = performance.now();

  const probabilityName =
    active.session.outputNames.find((name) => name !== active.session.outputNames[0]) ??
    active.session.outputNames[0]!;
  const raw = output[probabilityName]!.data as Float32Array;
  const posteriors = Array.from(raw);

  const decision = active.accumulator.update(posteriors);

  const channelRms: number[] = [];
  for (const channel of window) {
    let sum = 0;
    for (let i = 0; i < channel.length; i++) sum += channel[i]! * channel[i]!;
    channelRms.push(Math.sqrt(sum / channel.length));
  }

  const display = decimate(window, DISPLAY_POINTS);

  const message: DecisionResponse = {
    type: 'decision',
    leader: decision.leader,
    commitment: decision.commitment,
    moving: decision.moving,
    reversible: decision.reversible,
    latched: decision.latched,
    latchedClass: decision.latchedClass,
    timedOut: decision.timedOut,
    hops: decision.hops,
    effectiveThreshold: decision.effectiveThreshold,
    evidence: decision.evidence,
    posteriors,
    channelRms,
    display,
    latencyMs: inferenceEnd - featureStart,
    featureMs: featureEnd - featureStart,
    inferenceMs: inferenceEnd - featureEnd,
  };

  post(message, [display.buffer]);
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  try {
    switch (request.type) {
      case 'init':
        await init(
          request.modelUrl,
          request.nChannels,
          request.samplingRateHz,
          request.risks,
        );
        return;

      case 'reset':
        runtime?.accumulator.reset();
        return;

      case 'setRisks':
        if (runtime) {
          runtime.accumulator = new EvidenceAccumulator(request.risks);
        }
        return;

      case 'samples': {
        const active = runtime;
        if (!active) return;

        const perChannel = active.nChannels;
        const count = request.samples.length / perChannel;
        const channels: Float64Array[] = [];
        for (let c = 0; c < perChannel; c++) {
          const view = new Float64Array(count);
          for (let i = 0; i < count; i++) view[i] = request.samples[c * count + i]!;
          channels.push(view);
        }
        active.ring.push(channels);

        // Decode on exact hop boundaries. Decoding per message would tie the
        // decision rate to however the source happens to chunk its output.
        active.pending += count;
        while (active.pending >= active.hop) {
          active.pending -= active.hop;
          await decodeWindow(active);
        }
        return;
      }
    }
  } catch (error) {
    fail(error);
  }
};
