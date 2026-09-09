import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@neurogrip/design/tokens.css';
import './app.css';
import { App } from './App.js';

const container = document.getElementById('root');
if (!container) throw new Error('#root not found');

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
