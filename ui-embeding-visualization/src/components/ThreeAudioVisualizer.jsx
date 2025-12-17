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

  useEffect(() => {
    if (!containerRef.current) return;
    
    const container = containerRef.current;

    // Initialize Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a1a);
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
    canvas.width = 512;
    canvas.height = 256;
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
        gradient.addColorStop(0, '#0a0a1a');
        gradient.addColorStop(0.5, '#0d0d25');
        gradient.addColorStop(1, '#0a0a1a');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw spectrum with circular/wave style for embedded view
        const barWidth = canvas.width / dataArray.length;
        const centerY = canvas.height / 2;
        
        for (let i = 0; i < dataArray.length; i++) {
          const barHeight = (dataArray[i] / 255) * (canvas.height * 0.45);
          
          // Create color based on frequency intensity with cyan/magenta theme
          const hue = 180 + (i / dataArray.length) * 60; // Cyan to blue range
          const intensity = dataArray[i] / 255;
          
          // Draw mirrored bars from center
          const gradient2 = ctx.createLinearGradient(0, centerY - barHeight, 0, centerY + barHeight);
          gradient2.addColorStop(0, `hsla(${hue}, 100%, ${intensity * 60 + 20}%, 0.9)`);
          gradient2.addColorStop(0.5, `hsla(${hue + 30}, 100%, ${intensity * 70 + 30}%, 1)`);
          gradient2.addColorStop(1, `hsla(${hue}, 100%, ${intensity * 60 + 20}%, 0.9)`);
          
          ctx.fillStyle = gradient2;
          ctx.fillRect(i * barWidth, centerY - barHeight, barWidth - 1, barHeight * 2);
        }

        // Add glow effect
        ctx.shadowBlur = 15;
        ctx.shadowColor = 'rgba(0, 255, 255, 0.3)';

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
        setIsPlaying(false);
      });

      // Play audio
      await audio.play();
      setIsPlaying(true);
    } catch (err) {
      console.error('Error playing audio:', err);
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

  // Always render as embedded panel
  return (
    <div style={{
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(0, 0, 0, 0.95)',
      borderRadius: '8px',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      border: '1px solid rgba(0, 255, 255, 0.25)',
      boxShadow: '0 4px 20px rgba(0, 255, 255, 0.1)'
    }}>
      {/* Compact Header */}
      <div style={{
        padding: '10px 12px',
        color: 'white',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: 'rgba(0, 20, 40, 0.8)',
        borderBottom: '1px solid rgba(0, 255, 255, 0.2)'
      }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ 
            fontSize: '13px', 
            fontWeight: '600', 
            marginBottom: '2px',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            color: '#00ffff'
          }}>
            {songName}
          </div>
          <div style={{ fontSize: '11px', color: '#888' }}>
            {genre}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
          <button
            onClick={isPlaying ? stopAudio : playAudio}
            style={{
              width: '32px',
              height: '32px',
              backgroundColor: isPlaying ? 'rgba(255, 80, 80, 0.8)' : 'rgba(0, 255, 255, 0.8)',
              color: isPlaying ? 'white' : '#000',
              border: 'none',
              borderRadius: '50%',
              cursor: 'pointer',
              fontSize: '14px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s'
            }}
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
          <button
            onClick={handleClose}
            style={{
              width: '32px',
              height: '32px',
              backgroundColor: 'rgba(100, 100, 100, 0.5)',
              color: 'white',
              border: 'none',
              borderRadius: '50%',
              cursor: 'pointer',
              fontSize: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s'
            }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* Visualization container */}
      <div
        ref={containerRef}
        style={{
          flex: 1,
          width: '100%',
          position: 'relative',
          minHeight: '80px'
        }}
      />
    </div>
  );
};

