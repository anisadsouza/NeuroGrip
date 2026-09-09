import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { NEUROGRIP_ALIASES } from './aliases.js';

export default defineConfig({
  plugins: [react()],
  resolve: { alias: NEUROGRIP_ALIASES },
  worker: { format: 'es' },
  build: {
    rollupOptions: {
      // The bench is a second entry, not a route: it drives the real worker
      // outside React, so the path it measures is the shipped one.
      input: {
        app: fileURLToPath(new URL('index.html', import.meta.url)),
        bench: fileURLToPath(new URL('bench.html', import.meta.url)),
      },
    },
  },
  server: {
    // ONNX Runtime Web needs cross-origin isolation to use SharedArrayBuffer
    // for multi-threaded WASM. Without these headers it silently falls back to
    // single-threaded, which roughly doubles inference latency.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    // Vite currently inherits server.headers into preview, so this is a pin
    // rather than a fix: it keeps cross-origin isolation if that inheritance
    // ever changes, and it is what the Playwright bench runs against.
    //
    // Neither block helps a static host. Azure Static Web Apps, GitHub Pages
    // and any plain file server need their own header configuration, which is
    // a deployment concern (Phase 5). The app reports which mode it got, so
    // the degradation is visible rather than silent.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
