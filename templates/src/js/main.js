import { createBloodDrops } from './bloodDrops.js';
import { FingerprintCapture } from './fingerprintCapture.js';

document.addEventListener('DOMContentLoaded', () => {
  createBloodDrops();
  new FingerprintCapture();
});