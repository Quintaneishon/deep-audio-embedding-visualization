import h5py
import numpy as np
import torch
import json
import argparse
from pathlib import Path

def generate_stats_from_h5(h5_path, output_json_path):
    print(f"Loading features from {h5_path}...")
    
    all_features = []
    
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        print(f"Found {len(keys)} cached files.")
        
        for key in keys:
            feat = f[key][:]
            all_features.append(feat)
            
    all_features = np.array(all_features)
    print(f"Features shape: {all_features.shape}")
    
    # Compute stats
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    
    feature_names = [
        'spectral_centroid',
        'spectral_bandwidth', 
        'spectral_rolloff',
        'zero_crossing_rate',
        'rms_energy',
        'tempo'
    ]
    
    stats = {
        'feature_mean': mean.tolist(),
        'feature_std': std.tolist(),
        'feature_names': feature_names
    }
    
    # Save
    with open(output_json_path, 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"✓ Stats saved to {output_json_path}")
    
    # Print for verification
    print("\n computed stats:")
    for i, name in enumerate(feature_names):
        print(f"  {name:20s}: mean={mean[i]:.2f}, std={std[i]:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    generate_stats_from_h5(args.h5, args.output)

