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
  assertFeatureSpecCompatible,
  checkSignalQuality,
  featureVector,
  hopSamples,
  windowSamples,
} from '@neurogrip/core';
import { channelRms, decimate, deinterleave } from './frames.js';
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
  // crossOriginIsolated, not `typeof SharedArrayBuffer`. The constructor can
  // exist while isolation is absent, in which case the threads never
  // materialise and the readout would claim 'enabled' over single-threaded
  // inference -- an optimistic number, which is the one thing this readout
  // exists to avoid.
  const threaded = self.crossOriginIsolated === true;
  ort.env.wasm.numThreads = threaded
    ? Math.min(4, navigator.hardwareConcurrency || 1)
    : 1;
  ort.env.wasm.simd = true;
  // Respecting BASE_URL, as the model and replay URLs already do. Hard-coding
  // the origin root breaks the moment the app is served from a subpath.
  ort.env.wasm.wasmPaths = new URL(`${import.meta.env.BASE_URL}ort/`, self.location.href).href;

  const metadataUrl = modelUrl.replace(/\.onnx$/, '.json');
  const metadata = await fetch(metadataUrl).then((r) => r.json());

  // Before the session loads, not after. A decoder exported under a different
  // feature specification runs perfectly happily on a vector whose columns
  // mean something else, and the only evidence is degraded accuracy.
  assertFeatureSpecCompatible({
    source: 'decoder.json',
    featureSpecVersion: metadata.feature_spec_version,
    nChannels: metadata.n_channels,
    nFeatures: metadata.n_features,
  });

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

  const rms = channelRms(window);

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
    channelRms: rms,
    display,
    latencyMs: inferenceEnd - featureStart,
    featureMs: featureEnd - featureStart,
    inferenceMs: inferenceEnd - featureEnd,
  };

  post(message, [display.buffer]);
}

/**
 * Messages are handled strictly one at a time.
 *
 * `MultiChannelRingBuffer.readLatestWindow` returns views into a shared
 * scratch buffer, invalidated by the next call. An async handler makes the
 * worker concurrent by accident: a second `samples` message arriving while the
 * first is awaiting `session.run` reads the ring again, overwrites the scratch,
 * and the first decode then computes its RMS and its display trace from the
 * wrong window. It is reachable in ordinary use -- after a frame stall the
 * replay source delivers up to a quarter second of samples, which is twelve
 * hops of sequential decoding and longer than one frame -- and the symptom is
 * a garbled trace and a wrong drive colour with no error anywhere.
 *
 * Chaining rather than copying: this is a serial pipeline, and the `async`
 * keyword was the only thing that ever made it otherwise.
 */
let queue: Promise<void> = Promise.resolve();

async function handle(request: WorkerRequest): Promise<void> {
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

        const channels = deinterleave(request.samples, active.nChannels);
        const count = channels[0]?.length ?? 0;
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
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  queue = queue.then(() => handle(event.data));
};
