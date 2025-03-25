import React, { useState } from 'react';
import { Droplet } from 'lucide-react';

function App() {
  const [status, setStatus] = useState('');
  const [showResult, setShowResult] = useState(false);

  const handleCapture = () => {
    setStatus('Place Finger...');
    setTimeout(() => setStatus('Finger Detected...'), 1000);
    setTimeout(() => setStatus('Capturing...'), 2000);
    setTimeout(() => {
      setStatus('Predicting Blood Group...');
      setShowResult(true);
    }, 3000);
  };

  return (
    <div className="app-container">
      <div className="blood-drops">
        {[...Array(20)].map((_, i) => (
          <div key={i} className="drop" style={{ '--delay': `${i * 0.5}s` } as React.CSSProperties} />
        ))}
      </div>
      
      <div className="content">
        <h1>Blood Group Prediction</h1>
        <div className="capture-section">
          <button className="capture-button" onClick={handleCapture} disabled={!!status}>
            <Droplet className="icon" />
            Capture Fingerprint
          </button>
          {status && <div className="status-message">{status}</div>}
        </div>

        {showResult && (
          <div className="result-container">
            <div className="fingerprint-display">
              <h3>Fingerprint Image</h3>
              <div className="fingerprint-placeholder" />
            </div>
            <div className="prediction">
              <h4>Predicted Blood Group:</h4>
              <span className="blood-group">A+</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;