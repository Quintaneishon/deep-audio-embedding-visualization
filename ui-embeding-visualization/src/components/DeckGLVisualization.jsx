import React, { useState, useEffect, useMemo, useRef } from 'react';
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
  onRefetchEmbeddings,
}) => {
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [showVisualizer, setShowVisualizer] = useState(false);
  const [currentAudioUrl, setCurrentAudioUrl] = useState(null);
  const [selectedPointInfo, setSelectedPointInfo] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [blinkingPoint, setBlinkingPoint] = useState(null);
  const [blinkVisible, setBlinkVisible] = useState(true); // Controls blink on/off state
  const blinkingRef = useRef(null); // Track the blinking point for the interval
  const dataZoomRef = useRef(4); // Store the appropriate zoom level for the data scale
  const [isUploading, setIsUploading] = useState(false);
  const [toast, setToast] = useState(null);
  const fileInputRef = useRef(null);

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

      // Store the target zoom for navigation
      dataZoomRef.current = targetZoom;

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

  // Filter songs based on search query
  const filteredSongs = useMemo(() => {
    if (!searchQuery.trim()) return [];
    
    const query = searchQuery.toLowerCase();
    return points.filter(point => 
      point.name.toLowerCase().includes(query) ||
      point.tag.toLowerCase().includes(query)
    ).slice(0, 50); // Limit to 50 results for performance
  }, [searchQuery, points]);

  // Clear blinking marker when search query changes
  useEffect(() => {
    if (searchQuery) {
      setBlinkingPoint(null);
    }
  }, [searchQuery]);

  // Animate the blinking effect by toggling hover on/off
  useEffect(() => {
    if (!blinkingPoint) {
      blinkingRef.current = null;
      setBlinkVisible(false);
      return;
    }
    
    blinkingRef.current = blinkingPoint;
    let isOn = true;
    setBlinkVisible(true);
    
    const interval = setInterval(() => {
      if (blinkingRef.current) {
        isOn = !isOn;
        setBlinkVisible(isOn);
        setHoveredPoint(isOn ? blinkingRef.current : null);
      }
    }, 600); // Toggle every 600ms for slower, easier to click blinking
    
    return () => clearInterval(interval);
  }, [blinkingPoint]);

  // Show toast notification
  const showToast = (message, type = 'error') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 5000); // Auto-dismiss after 5 seconds
  };

  // Handle file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.mp3')) {
      showToast('Only MP3 files are supported', 'error');
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('Uploading file:', file.name);
      const response = await fetch('http://127.0.0.1:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      console.log('Upload response:', result);

      // Check if the response was successful (status 200-299)
      if (!response.ok) {
        console.error('Upload failed with status:', response.status);
        showToast(result.error || `Upload failed with status ${response.status}`, 'error');
        return;
      }

      if (result.success) {
        showToast('Song uploaded successfully!', 'success');
        
        // Refetch embeddings
        if (onRefetchEmbeddings) {
          console.log('Refetching embeddings...');
          const newEmbeddings = await onRefetchEmbeddings();
          
          // Find the newly added song by filename
          if (newEmbeddings && newEmbeddings.length > 0) {
            const newSong = newEmbeddings.find(emb => emb.audio === result.filename);
            if (newSong) {
              console.log('Found new song:', newSong.name);
              // Transform to point format and navigate
              const newPoint = {
                position: dimensiones === 3 
                  ? [newSong.coords[0], newSong.coords[1], newSong.coords[2] || 0]
                  : [newSong.coords[0], newSong.coords[1], 0],
                color: getColorForGenre(newSong.tag),
                name: newSong.name,
                tag: newSong.tag,
                audio: newSong.audio,
                id: `${newSong.name}_new`
              };
              
              // Navigate to the new song
              setTimeout(() => {
                navigateToPoint(newPoint);
              }, 500);
            } else {
              console.warn('Could not find newly uploaded song in embeddings');
            }
          }
        }
      } else {
        console.error('Upload result indicated failure:', result);
        showToast(result.error || 'Failed to upload file', 'error');
      }
    } catch (error) {
      console.error('Upload error:', error);
      showToast(`Error: ${error.message}`, 'error');
    } finally {
      setIsUploading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Navigate to a specific point without playing audio
  const navigateToPoint = (point) => {
    setViewState(prev => ({
      ...prev,
      target: point.position,
      zoom: dataZoomRef.current + 4, // Zoom in much closer to the point
      transitionDuration: 1000 // Smooth 1 second animation
    }));
    
    // Set blinking marker (stays until user clicks a point or searches again)
    setBlinkingPoint(point);
    setHoveredPoint(point);
    
    console.log('Navigated to:', point.name);
  };

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
    }),
    // Blinking marker layer - shows a large bright marker at the searched point
    blinkingPoint && blinkVisible && new PointCloudLayer({
      id: 'blinking-marker-layer',
      data: [blinkingPoint],
      getPosition: d => d.position,
      getColor: [255, 255, 0], // Bright yellow
      pointSize: 9, // Much larger than regular points
      pickable: true,
      onClick: (info) => {
        if (info.object) {
          const point = info.object;
          console.log('Clicked blinking marker:', point.name, point.tag);
          if (point.audio) {
            const audioUrl = `http://127.0.0.1:5000/audio/${point.audio}`;
            setCurrentAudioUrl(audioUrl);
            setSelectedPointInfo({ name: point.name, tag: point.tag, position: point.position });
            setShowVisualizer(true);
          }
        }
      }
    }),
    // Outer ring effect for blinking marker
    blinkingPoint && !blinkVisible && new PointCloudLayer({
      id: 'blinking-marker-ring-layer',
      data: [blinkingPoint],
      getPosition: d => d.position,
      getColor: [255, 100, 0], // Orange when "off"
      pointSize: 8,
      pickable: true,
      onClick: (info) => {
        if (info.object) {
          const point = info.object;
          console.log('Clicked blinking marker:', point.name, point.tag);
          if (point.audio) {
            const audioUrl = `http://127.0.0.1:5000/audio/${point.audio}`;
            setCurrentAudioUrl(audioUrl);
            setSelectedPointInfo({ name: point.name, tag: point.tag, position: point.position });
            setShowVisualizer(true);
          }
        }
      }
    })
  ].filter(Boolean);

  return (
    <div className="deckgl-container">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".mp3"
        onChange={handleFileUpload}
        style={{ display: 'none' }}
      />

      {/* Loading Overlay */}
      {isUploading && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 100,
          color: 'white'
        }}>
          <div style={{
            width: '50px',
            height: '50px',
            border: '5px solid rgba(255, 255, 255, 0.3)',
            borderTop: '5px solid white',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <p style={{ marginTop: '20px', fontSize: '16px' }}>
            Processing audio file...
          </p>
          <p style={{ marginTop: '8px', fontSize: '12px', opacity: 0.7 }}>
            This may take a few minutes
          </p>
        </div>
      )}

      {/* Toast Notification */}
      {toast && (
        <div style={{
          position: 'absolute',
          top: '70px',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: toast.type === 'error' ? 'rgba(220, 38, 38, 0.95)' : 'rgba(34, 197, 94, 0.95)',
          color: 'white',
          padding: '12px 24px',
          borderRadius: '8px',
          fontSize: '14px',
          zIndex: 1000,
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
          maxWidth: '400px',
          textAlign: 'center'
        }}>
          {toast.message}
        </div>
      )}

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

      {/* Embedded Audio Visualizer Panel - Bottom Right */}
      {showVisualizer && currentAudioUrl && selectedPointInfo && (
        <div className="visualizer-panel" style={{
          position: 'absolute',
          bottom: 10,
          right: 10,
          width: '340px',
          height: '180px',
          backgroundColor: 'rgba(0, 0, 0, 0.9)',
          borderRadius: '8px',
          zIndex: 1,
          overflow: 'hidden',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)'
        }}>
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
        </div>
      )}

      {/* Legend */}
      <div className="legend" style={{
        position: 'absolute',
        top: 60,
        right: 10,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '12px',
        borderRadius: '4px',
        fontSize: '12px',
        maxHeight: 'calc(100vh - 220px)',
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

      {/* Search Panel */}
      <div className="search-panel" style={{
        position: 'absolute',
        bottom: 10,
        left: 10,
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        color: 'white',
        padding: '12px',
        borderRadius: '8px',
        fontSize: '13px',
        zIndex: 1,
        width: '320px',
        maxHeight: '400px',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)'
      }}>
        {/* Upload Button - Top Right of Search Panel */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          style={{
            position: 'absolute',
            top: '-18px',
            right: '-18px',
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            backgroundColor: isUploading ? 'rgba(100, 100, 100, 0.9)' : 'rgba(0, 0, 0, 0.9)',
            border: '2px solid rgba(255, 255, 255, 0.3)',
            color: 'white',
            fontSize: '22px',
            fontWeight: 'bold',
            cursor: isUploading ? 'not-allowed' : 'pointer',
            zIndex: 10,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)'
          }}
          onMouseEnter={(e) => {
            if (!isUploading) {
              e.currentTarget.style.backgroundColor = 'rgba(50, 50, 50, 0.9)';
              e.currentTarget.style.transform = 'scale(1.05)';
            }
          }}
          onMouseLeave={(e) => {
            if (!isUploading) {
              e.currentTarget.style.backgroundColor = 'rgba(0, 0, 0, 0.9)';
              e.currentTarget.style.transform = 'scale(1)';
            }
          }}
          title="Upload MP3 file"
        >
          {isUploading ? '...' : '+'}
        </button>

        <div style={{ marginBottom: '8px', fontWeight: 'bold', fontSize: '14px' }}>
        Search Songs
        </div>
        
        <input
          type="text"
          placeholder="Type song name or genre..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            padding: '8px 12px',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '4px',
            color: 'white',
            fontSize: '13px',
            outline: 'none',
            marginBottom: '8px'
          }}
        />
        
        {searchQuery && (
          <div style={{
            fontSize: '11px',
            opacity: 0.7,
            marginBottom: '8px'
          }}>
            {filteredSongs.length} {filteredSongs.length === 1 ? 'result' : 'results'} found
          </div>
        )}
        
        {searchQuery && filteredSongs.length > 0 && (
          <div style={{
            overflowY: 'auto',
            maxHeight: '280px',
            marginTop: '4px'
          }}>
            {filteredSongs.map((point, index) => (
              <div
                key={point.id}
                onClick={() => navigateToPoint(point)}
                style={{
                  padding: '8px',
                  marginBottom: '4px',
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  border: '1px solid transparent'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.15)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.3)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
                  e.currentTarget.style.borderColor = 'transparent';
                }}
              >
                <div style={{ fontWeight: '500', marginBottom: '2px' }}>
                  {point.name}
                </div>
                <div style={{
                  fontSize: '11px',
                  opacity: 0.7,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}>
                  <span style={{
                    display: 'inline-block',
                    width: '8px',
                    height: '8px',
                    backgroundColor: `rgb(${point.color[0]}, ${point.color[1]}, ${point.color[2]})`,
                    borderRadius: '50%'
                  }} />
                  {point.tag}
                </div>
              </div>
            ))}
          </div>
        )}
        
        {searchQuery && filteredSongs.length === 0 && (
          <div style={{
            padding: '16px',
            textAlign: 'center',
            opacity: 0.6,
            fontSize: '12px'
          }}>
            No songs found matching "{searchQuery}"
          </div>
        )}
      </div>
    </div>
  );
};

