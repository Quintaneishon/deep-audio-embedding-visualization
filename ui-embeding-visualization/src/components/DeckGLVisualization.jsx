import React, { useState, useEffect, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { PointCloudLayer } from '@deck.gl/layers';
import { OrbitView } from '@deck.gl/core';
import { ThreeAudioVisualizer } from './ThreeAudioVisualizer';
import '../styles/deckgl.css';

const INITIAL_VIEW_STATE = {
  target: [0, 0, 0],
  rotationX: 0,
  rotationOrbit: 0,
  zoom: 0,
  minZoom: -10,
  maxZoom: 10
};

// Color mapping for different genres/tags
const GENRE_COLORS = {
  'rock': [255, 0, 0],
  'pop': [255, 105, 180],
  'classical': [138, 43, 226],
  'jazz': [255, 215, 0],
  'electronic': [0, 255, 255],
  'hip-hop': [255, 140, 0],
  'country': [139, 69, 19],
  'blues': [0, 0, 255],
  'metal': [128, 128, 128],
  'folk': [34, 139, 34],
  'default': [200, 200, 200]
};

// Function to get color for a genre
const getColorForGenre = (genre) => {
  if (!genre) return GENRE_COLORS.default;
  const lowerGenre = genre.toLowerCase();
  return GENRE_COLORS[lowerGenre] || GENRE_COLORS.default;
};

export const DeckGLVisualization = ({
  embeddings,
  dimensiones,
  agruparPor
}) => {
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [showVisualizer, setShowVisualizer] = useState(false);
  const [currentAudioUrl, setCurrentAudioUrl] = useState(null);
  const [selectedPointInfo, setSelectedPointInfo] = useState(null);

  // Log when data loads
  useEffect(() => {
    if (embeddings.length > 0) {
      console.log(`Loaded ${embeddings.length} audio embedding points`);
    }
  }, [embeddings]);

  // Transform embeddings data for deck.gl
  const points = useMemo(() => {
    return embeddings.map((embedding, index) => ({
      position: dimensiones === 3 
        ? [embedding.coords[0], embedding.coords[1], embedding.coords[2] || 0]
        : [embedding.coords[0], embedding.coords[1], 0],
      color: getColorForGenre(embedding.tag),
      name: embedding.name,
      tag: embedding.tag,
      audio: embedding.audio,
      id: `${embedding.name}_${index}`
    }));
  }, [embeddings, dimensiones]);

  // Calculate the center of the point cloud to properly position the camera
  useEffect(() => {
    if (points.length > 0) {
      const sum = points.reduce((acc, point) => ({
        x: acc.x + point.position[0],
        y: acc.y + point.position[1],
        z: acc.z + point.position[2]
      }), { x: 0, y: 0, z: 0 });

      const center = [
        sum.x / points.length,
        sum.y / points.length,
        sum.z / points.length
      ];

      // Calculate the spread/scale of the data to set appropriate zoom
      const distances = points.map(p => {
        const dx = p.position[0] - center[0];
        const dy = p.position[1] - center[1];
        const dz = p.position[2] - center[2];
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
      });
      const maxDistance = Math.max(...distances);
      
      // Set zoom based on data scale (larger spread = zoom out more)
      const baseZoom = Math.max(0, 5 - Math.log2(maxDistance));
      const targetZoom = baseZoom + 4;

      console.log('Point cloud center:', center);
      console.log('Point cloud max distance:', maxDistance);
      console.log('Base zoom level:', baseZoom);
      console.log('Target zoom level:', targetZoom);
      console.log('Total points:', points.length);

      // Start at base zoom, then animate to target zoom
      setViewState(prev => ({
        ...prev,
        target: center,
        zoom: baseZoom,
        transitionDuration: 0
      }));

      // Animate zoom in after a short delay
      setTimeout(() => {
        setViewState(prev => ({
          ...prev,
          zoom: targetZoom,
          transitionDuration: 2000 // 2 second smooth zoom animation
        }));
      }, 100);
    }
  }, [points]);

  // Handle keyboard events (ESC to close visualizer)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        if (showVisualizer) {
          setShowVisualizer(false);
          setCurrentAudioUrl(null);
          setSelectedPointInfo(null);
          console.log("Visualizer closed with ESC");
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [showVisualizer]);

  const layers = [
    new PointCloudLayer({
      id: 'point-cloud-layer',
      data: points,
      getPosition: d => d.position,
      getColor: d => d.color,
      pointSize: 2,
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 0, 200],
      onHover: (info) => {
        if (info.object) {
          setHoveredPoint(info.object);
        } else {
          setHoveredPoint(null);
        }
      },
      onClick: (info) => {
        if (info.object) {
          const point = info.object;
          console.log('Clicked:', point.name, point.tag);
          
          // Open Three.js audio visualizer
          if (point.audio) {
            const audioUrl = `http://127.0.0.1:5000/audio/${point.audio}`;
            console.log("Opening visualizer for:", audioUrl);
            
            setCurrentAudioUrl(audioUrl);
            setSelectedPointInfo({
              name: point.name,
              tag: point.tag,
              position: point.position
            });
            setShowVisualizer(true);
          }
        }
      }
    })
  ];

  return (
    <div className="deckgl-container">
      <DeckGL
        views={new OrbitView()}
        viewState={viewState}
        onViewStateChange={({ viewState }) => setViewState(viewState)}
        controller={true}
        layers={layers}
        parameters={{
          clearColor: [0.07, 0.07, 0.07, 1]
        }}
      >
        {hoveredPoint && (
          <div className="tooltip" style={{
            position: 'absolute',
            zIndex: 1,
            pointerEvents: 'none',
            left: '50%',
            top: 10,
            transform: 'translateX(-50%)',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '14px'
          }}>
            <div><strong>{hoveredPoint.name}</strong></div>
            <div>Genre: {hoveredPoint.tag}</div>
            <div style={{ fontSize: '12px', marginTop: '4px', opacity: 0.8 }}>
              Click to visualize audio
            </div>
          </div>
        )}
      </DeckGL>

      {/* Three.js Audio Visualizer */}
      {showVisualizer && currentAudioUrl && selectedPointInfo && (
        <ThreeAudioVisualizer
          audioUrl={currentAudioUrl}
          songName={selectedPointInfo.name}
          genre={selectedPointInfo.tag}
          coordinates={selectedPointInfo.position}
          onClose={() => {
            setShowVisualizer(false);
            setCurrentAudioUrl(null);
            setSelectedPointInfo(null);
          }}
        />
      )}

      {/* Legend */}
      <div className="legend" style={{
        position: 'absolute',
        top: 80,
        right: 10,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '12px',
        borderRadius: '4px',
        fontSize: '12px',
        maxHeight: '80vh',
        overflowY: 'auto',
        zIndex: 1
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>Genres</div>
        {Object.entries(GENRE_COLORS).filter(([key]) => key !== 'default').map(([genre, color]) => (
          <div key={genre} style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <div style={{
              width: '12px',
              height: '12px',
              backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
              marginRight: '8px',
              borderRadius: '2px'
            }} />
            <span>{genre}</span>
          </div>
        ))}
      </div>

      {/* Instructions */}
      <div className="instructions" style={{
        position: 'absolute',
        bottom: 10,
        left: 10,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '10px',
        borderRadius: '4px',
        fontSize: '11px',
        zIndex: 1
      }}>
        <div><strong>Controls:</strong></div>
        <div>• Left click + drag: Rotate</div>
        <div>• Right click + drag: Pan</div>
        <div>• Scroll: Zoom</div>
        <div>• Click point: Open visualizer</div>
        <div>• ESC: Close visualizer</div>
      </div>
    </div>
  );
};

