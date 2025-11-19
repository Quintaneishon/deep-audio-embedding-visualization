/**
 * Generates a distinct color for each genre using HSL color space
 * This ensures good color distribution and visibility
 */
export const generateColorPalette = (genres) => {
    const colorMap = {};
    
    // Golden ratio conjugate for better distribution
    const goldenRatio = 0.618033988749895;
    let hue = Math.random(); // Start with random hue for variety
    
    genres.forEach((genre, index) => {
        // Use golden ratio to distribute hues evenly
        hue = (hue + goldenRatio) % 1;
        
        // Vary saturation and lightness for more distinct colors
        const saturation = 65 + (index % 3) * 10; // 65%, 75%, 85%
        const lightness = 60 + (index % 2) * 10;   // 60%, 70%
        
        const h = Math.floor(hue * 360);
        const s = saturation;
        const l = lightness;
        
        colorMap[genre] = hslToHex(h, s, l);
    });
    
    return colorMap;
};

/**
 * Extracts unique genres from data array
 */
export const extractGenresFromData = (data) => {
    const genresSet = new Set();
    
    data.forEach(item => {
        if (item.tag) {
            const match = item.tag.match(/genre---(.+)/);
            if (match) {
                genresSet.add(match[1]);
            }
        }
    });
    
    return Array.from(genresSet).sort();
};

/**
 * Extracts genre from a tag string
 */
export const extractGenre = (tag) => {
    if (!tag) return 'default';
    const match = tag.match(/genre---(.+)/);
    return match ? match[1] : 'default';
};

/**
 * Converts HSL to Hex color
 */
function hslToHex(h, s, l) {
    s /= 100;
    l /= 100;
    
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;
    
    let r = 0, g = 0, b = 0;
    
    if (0 <= h && h < 60) {
        r = c; g = x; b = 0;
    } else if (60 <= h && h < 120) {
        r = x; g = c; b = 0;
    } else if (120 <= h && h < 180) {
        r = 0; g = c; b = x;
    } else if (180 <= h && h < 240) {
        r = 0; g = x; b = c;
    } else if (240 <= h && h < 300) {
        r = x; g = 0; b = c;
    } else if (300 <= h && h < 360) {
        r = c; g = 0; b = x;
    }
    
    r = Math.round((r + m) * 255);
    g = Math.round((g + m) * 255);
    b = Math.round((b + m) * 255);
    
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function toHex(n) {
    const hex = n.toString(16);
    return hex.length === 1 ? '0' + hex : hex;
}

/**
 * Predefined palette for common genres (fallback)
 */
export const PREDEFINED_COLORS = {
    'ambient': '#8dd3c7',
    'hiphop': '#ffffb3',
    'rock': '#bebada',
    'electronic': '#fb8072',
    'jazz': '#80b1d3',
    'classical': '#fdb462',
    'pop': '#b3de69',
    'metal': '#fccde5',
    'folk': '#d9d9d9',
    'country': '#bc80bd',
    'blues': '#ccebc5',
    'reggae': '#ffed6f',
    'soul': '#ff9999',
    'punk': '#cc99ff',
    'indie': '#99ccff',
    'default': '#999999'
};

/**
 * Gets color for a genre, using predefined colors first, then generating if needed
 */
export const getGenreColor = (genre, genreColorMap, usePredefined = true) => {
    if (usePredefined && PREDEFINED_COLORS[genre]) {
        return PREDEFINED_COLORS[genre];
    }
    return genreColorMap[genre] || PREDEFINED_COLORS['default'];
};

