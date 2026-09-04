/**
 * The contract between the UI thread and the inference worker.
 *
 * Everything expensive happens in the worker: windowing, feature extraction,
 * ONNX inference, and evidence accumulation. The UI thread receives a small
 * decision object roughly every 20 ms and does nothing but draw. That split is
 * what keeps the interface responsive while the decoder runs at 50 Hz.
 *
 * On privacy: the project's requirement is that raw sEMG is never persisted and
 * never leaves the device. Samples do cross this thread boundary, because the
 * oscilloscope has to draw the wearer's own signal on the wearer's own screen.
 * What is forbidden is writing it to storage or sending it over the network,
 * and neither happens anywhere in this application.
 */

/**
 * Points per channel sent for display.
 *
 * 240 across a 200 ms window is 1.2 points per millisecond, which resolves the
 * ~100 Hz content that dominates surface EMG. At 120 the trace aliased into
 * smooth undulation and stopped looking like the signal it is. The cost is
 * 7.7 KB per message at 50 Hz, sent as a transferable, which is nothing.
 */
export const DISPLAY_POINTS = 240;

export interface InitRequest {
  readonly type: 'init';
  readonly modelUrl: string;
  readonly nChannels: number;
  readonly samplingRateHz: number;
  /** Commit cost per gesture, indexed like the model's classes. */
  readonly risks: readonly number[];
}

export interface SamplesRequest {
  readonly type: 'samples';
  /** Interleaved channel-major samples, in volts. */
  readonly samples: Float32Array;
  readonly nChannels: number;
}

export interface ResetRequest {
  readonly type: 'reset';
}

export interface SetRisksRequest {
  readonly type: 'setRisks';
  readonly risks: readonly number[];
}

export type WorkerRequest =
  | InitRequest
  | SamplesRequest
  | ResetRequest
  | SetRisksRequest;

export interface ReadyResponse {
  readonly type: 'ready';
  readonly gestures: readonly string[];
  readonly nFeatures: number;
  /** Which ONNX Runtime backend actually loaded, for the latency panel. */
  readonly backend: string;
  readonly threaded: boolean;
  readonly featureSpecVersion: number;
}

export interface DecisionResponse {
  readonly type: 'decision';
  readonly leader: number;
  readonly commitment: number;
  readonly moving: boolean;
  readonly reversible: boolean;
  readonly latched: boolean;
  readonly latchedClass: number | null;
  readonly timedOut: boolean;
  readonly hops: number;
  readonly effectiveThreshold: number;
  readonly evidence: readonly number[];
  readonly posteriors: readonly number[];
  /** Per-channel RMS in volts, for the activation ring. */
  readonly channelRms: readonly number[];
  /** Decimated window for the oscilloscope: nChannels x DISPLAY_POINTS. */
  readonly display: Float32Array;
  /** Feature extraction plus inference, in milliseconds, for this window. */
  readonly latencyMs: number;
  readonly featureMs: number;
  readonly inferenceMs: number;
}

export interface RejectedResponse {
  readonly type: 'rejected';
  readonly reason: string;
  readonly failedChannel: number | null;
}

export interface ErrorResponse {
  readonly type: 'error';
  readonly message: string;
}

export type WorkerResponse =
  | ReadyResponse
  | DecisionResponse
  | RejectedResponse
  | ErrorResponse;
