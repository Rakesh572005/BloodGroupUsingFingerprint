export function createBloodDrops() {
  const container = document.querySelector('.blood-drops');
  const dropCount = 20;

  for (let i = 0; i < dropCount; i++) {
    const drop = document.createElement('div');
    drop.className = 'drop';
    
    // Randomize drop properties
    const size = 8 + Math.random() * 8; // Random size between 8px and 16px
    drop.style.width = `${size}px`;
    drop.style.height = `${size * 1.3}px`; // Slightly taller than wide
    
    // Random position and delay
    drop.style.left = `${Math.random() * 100}%`;
    drop.style.animationDelay = `${Math.random() * 5}s`;
    drop.style.opacity = `${0.1 + Math.random() * 0.2}`; // Random opacity between 0.1 and 0.3
    
    // Random animation duration
    drop.style.animationDuration = `${6 + Math.random() * 4}s`; // Between 6s and 10s
    
    container.appendChild(drop);
  }
}