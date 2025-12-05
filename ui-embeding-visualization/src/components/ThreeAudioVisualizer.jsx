import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

export const ThreeAudioVisualizer = ({ audioUrl, songName, genre, coordinates, onClose }) => {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const analyserRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const animationFrameRef = useRef(null);
  const audioElementRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!containerRef.current) return;
    
    const container = containerRef.current;

    // Initialize Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000033);
    sceneRef.current = scene;

    // Setup orthographic camera for full-screen effect
    const aspect = container.clientWidth / container.clientHeight;
    const frustumSize = 2;
    const camera = new THREE.OrthographicCamera(
      frustumSize * aspect / -2,
      frustumSize * aspect / 2,
      frustumSize / 2,
      frustumSize / -2,
      0.1,
      1000
    );
    camera.position.z = 5;
    cameraRef.current = camera;

    // Setup renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Create canvas for spectrum visualization
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    
    // Create texture from canvas
    const texture = new THREE.CanvasTexture(canvas);
    
    // Create plane to display the spectrum (fill the viewport)
    const planeWidth = frustumSize * aspect;
    const planeHeight = frustumSize;
    const planeGeometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
    const planeMaterial = new THREE.MeshBasicMaterial({ 
      map: texture,
      side: THREE.DoubleSide
    });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    scene.add(plane);

    // Animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);

      if (analyserRef.current && ctx) {
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);

        // Clear canvas with gradient background
        const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
        gradient.addColorStop(0, '#000033');
        gradient.addColorStop(1, '#000011');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw spectrum
        const barWidth = canvas.width / dataArray.length;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
          const barHeight = (dataArray[i] / 255) * canvas.height;
          
          // Create color based on frequency intensity
          const hue = (i / dataArray.length) * 360;
          const intensity = dataArray[i] / 255;
          ctx.fillStyle = `hsl(${hue}, 100%, ${intensity * 50 + 25}%)`;
          
          // Draw bar from bottom
          ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
          
          x += barWidth;
        }

        // Update texture
        texture.needsUpdate = true;
      }

      renderer.render(scene, camera);
    };
    animate();

    // Handle window resize
    const handleResize = () => {
      if (!container) return;
      const newAspect = container.clientWidth / container.clientHeight;
      camera.left = frustumSize * newAspect / -2;
      camera.right = frustumSize * newAspect / 2;
      camera.top = frustumSize / 2;
      camera.bottom = frustumSize / -2;
      camera.updateProjectionMatrix();
      
      // Update plane size to match new aspect ratio
      const newPlaneWidth = frustumSize * newAspect;
      plane.scale.x = newPlaneWidth / planeWidth;
      
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (container && renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
      planeGeometry.dispose();
      planeMaterial.dispose();
      texture.dispose();
    };
  }, []);

  const playAudio = async () => {
    try {
      // Create audio context
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;

      // Create analyser
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      // Create audio element
      const audio = new Audio(audioUrl);
      audio.crossOrigin = "anonymous"; // Enable CORS for Web Audio API
      audioElementRef.current = audio;

      // Create media source
      const source = audioContext.createMediaElementSource(audio);
      sourceRef.current = source;

      // Connect audio graph
      source.connect(analyser);
      analyser.connect(audioContext.destination);

      // Add event listeners
      audio.addEventListener('ended', () => {
        setIsPlaying(false);
      });

      audio.addEventListener('error', (e) => {
        console.error('Audio error:', e);
        setError('Failed to load audio');
        setIsPlaying(false);
      });

      // Play audio
      await audio.play();
      setIsPlaying(true);
    } catch (err) {
      console.error('Error playing audio:', err);
      setError('Failed to play audio: ' + err.message);
      setIsPlaying(false);
    }
  };

  const stopAudio = () => {
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
    }
    setIsPlaying(false);
  };

  const handleClose = () => {
    stopAudio();
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(err => {
        console.log('AudioContext already closed:', err);
      });
    }
    onClose();
  };

  useEffect(() => {
    // Auto-play when component mounts
    let mounted = true;
    
    const initAudio = async () => {
      if (mounted) {
        await playAudio();
      }
    };
    
    initAudio();

    return () => {
      mounted = false;
      stopAudio();
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(err => {
          console.log('AudioContext already closed:', err);
        });
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl]);

  return (
    <div 
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backdropFilter: 'blur(5px)'
      }}
      onClick={handleClose}
    >
      {/* Popup Container */}
      <div 
        style={{
          width: '80%',
          maxWidth: '1000px',
          height: '80%',
          maxHeight: '700px',
          backgroundColor: '#000033',
          borderRadius: '12px',
          boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          border: '2px solid rgba(0, 255, 255, 0.3)'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{
          padding: '15px 20px',
          color: 'white',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          borderBottom: '1px solid rgba(0, 255, 255, 0.3)'
        }}>
          <div style={{ flex: 1 }}>
            <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 'bold', marginBottom: '5px' }}>
              {songName}
            </h2>
            <div style={{ fontSize: '13px', color: '#aaa', display: 'flex', gap: '15px' }}>
              <span><strong>Genre:</strong> {genre}</span>
              <span><strong>Position:</strong> ({coordinates[0].toFixed(2)}, {coordinates[1].toFixed(2)}{coordinates[2] !== 0 ? `, ${coordinates[2].toFixed(2)}` : ''})</span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            {isPlaying ? (
              <button
                onClick={stopAudio}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#ff4444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '13px',
                  fontWeight: '600',
                  transition: 'background-color 0.2s'
                }}
                onMouseEnter={(e) => e.target.style.backgroundColor = '#ff6666'}
                onMouseLeave={(e) => e.target.style.backgroundColor = '#ff4444'}
              >
                ⏸ Pause
              </button>
            ) : (
              <button
                onClick={playAudio}
                style={{
                  padding: '8px 16px',
                  backgroundColor: '#44ff44',
                  color: '#000',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '13px',
                  fontWeight: '600',
                  transition: 'background-color 0.2s'
                }}
                onMouseEnter={(e) => e.target.style.backgroundColor = '#66ff66'}
                onMouseLeave={(e) => e.target.style.backgroundColor = '#44ff44'}
              >
                ▶ Play
              </button>
            )}
            <button
              onClick={handleClose}
              style={{
                padding: '8px 16px',
                backgroundColor: '#666',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '13px',
                fontWeight: '600',
                transition: 'background-color 0.2s'
              }}
              onMouseEnter={(e) => e.target.style.backgroundColor = '#888'}
              onMouseLeave={(e) => e.target.style.backgroundColor = '#666'}
            >
              ✕ Close
            </button>
          </div>
        </div>

        {/* Visualization container */}
        <div
          ref={containerRef}
          style={{
            flex: 1,
            width: '100%',
            position: 'relative'
          }}
        />

        {/* Info overlay */}
        <div style={{
          padding: '12px 20px',
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          color: 'white',
          fontSize: '12px',
          textAlign: 'center',
          borderTop: '1px solid rgba(0, 255, 255, 0.3)'
        }}>
          {error ? (
            <span style={{ color: '#ff4444' }}>⚠️ {error}</span>
          ) : !isPlaying && (
            <span style={{ color: '#aaa' }}>Click Play to start</span>
          )}
        </div>
      </div>
    </div>
  );
};

