import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  resolve: {
    // Array form, and order matters: the subpath entry must be tried before the
    // bare package entry, or '@neurogrip/design/tokens.css' resolves to the
    // package index and Vite reports the file as missing.
    alias: [
      {
        find: '@neurogrip/design/tokens.css',
        replacement: fileURLToPath(new URL('../../packages/design/src/tokens.css', import.meta.url)),
      },
      {
        find: '@neurogrip/design',
        replacement: fileURLToPath(new URL('../../packages/design/src/index.ts', import.meta.url)),
      },
      {
        find: '@neurogrip/core',
        replacement: fileURLToPath(new URL('../../packages/core/src/index.ts', import.meta.url)),
      },
    ],
  },
  worker: { format: 'es' },
  server: {
    // ONNX Runtime Web needs cross-origin isolation to use SharedArrayBuffer
    // for multi-threaded WASM. Without these headers it silently falls back to
    // single-threaded, which roughly doubles inference latency.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
