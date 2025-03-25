export class FingerprintCapture {
  constructor() {
    this.status = '';
    this.captureButton = document.querySelector('.capture-button');
    this.statusMessage = document.querySelector('.status-message');
    this.resultContainer = document.querySelector('.result-container');
    this.fingerprintPlaceholder = document.querySelector('.fingerprint-placeholder');
    
    this.setupEventListeners();
  }

  setupEventListeners() {
    this.captureButton.addEventListener('click', () => this.handleCapture());
  }

  updateStatus(message) {
    this.status = message;
    this.statusMessage.textContent = message;
    this.statusMessage.style.display = message ? 'block' : 'none';
    this.captureButton.disabled = !!message;
  }

  showResult() {
    this.resultContainer.style.display = 'block';
    this.fingerprintPlaceholder.classList.add('captured');
  }

  async handleCapture() {
    const steps = [
      { 
        message: 'Place Finger...', 
        delay: 1000,
        action: () => this.fingerprintPlaceholder.classList.add('scanning')
      },
      { 
        message: 'Finger Detected...', 
        delay: 1000 
      },
      { 
        message: 'Capturing...', 
        delay: 1000,
        action: () => this.fingerprintPlaceholder.classList.remove('scanning')
      },
      { 
        message: 'Predicting Blood Group...', 
        delay: 1000 
      }
    ];

    for (const step of steps) {
      this.updateStatus(step.message);
      step.action?.();
      await new Promise(resolve => setTimeout(resolve, step.delay));
    }

    this.showResult();
    this.updateStatus('');
  }
}