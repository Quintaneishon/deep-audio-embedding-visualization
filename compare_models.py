import torch
import json

# Paths to the three models
checkpoint_paths = {
    "Whisper Contrastive": "/home/ar/Data/Ajitzi/deep-audio-embedding-visualization/ML/checkpoints/whisper_contrastive_20251128_085448/best_model.pth",
    "Lightweight Adapter (v1)": "/home/ar/Data/Ajitzi/deep-audio-embedding-visualization/ML/checkpoints/lightweight_adapter_20251217_142925/best_model.pth",
    "Lightweight Adapter (v2)": "/home/ar/Data/Ajitzi/deep-audio-embedding-visualization/ML/checkpoints/lightweight_adapter_20251218_050912/best_model.pth"
}

config_paths = {
    "Whisper Contrastive": "/home/ar/Data/Ajitzi/deep-audio-embedding-visualization/ML/checkpoints/whisper_contrastive_20251128_085448/training_config.json",
    "Lightweight Adapter (v1)": "/home/ar/Data/Ajitzi/deep-audio-embedding-visualization/ML/checkpoints/lightweight_adapter_20251217_142925/training_config.json",
    "Lightweight Adapter (v2)": "/home/ar/Data/Ajitzi/deep-audio-embedding-visualization/ML/checkpoints/lightweight_adapter_20251218_050912/training_config.json"
}

print("="*100)
print("MODEL COMPARISON - ALL THREE MODELS")
print("="*100)

checkpoints = {}
configs = {}

# Load all models and configs
for name, path in checkpoint_paths.items():
    print(f"\n📊 Loading {name}...")
    print("-" * 100)
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location='cpu')
    checkpoints[name] = checkpoint
    
    # Load config
    with open(config_paths[name], 'r') as f:
        config = json.load(f)
        configs[name] = config
    
    # Display basic info
    print(f"  Model Architecture:  {config.get('model_name', 'N/A')}")
    if 'projection_dim' in config:
        print(f"  Projection Dim:      {config['projection_dim']}")
    if 'feature_dim' in config:
        print(f"  Feature Dim:         {config['feature_dim']}")
    if 'output_dim' in config:
        print(f"  Output Dim:          {config['output_dim']}")
    if 'num_transformer_layers' in config:
        print(f"  Transformer Layers:  {config['num_transformer_layers']}")
    if 'finetune_early_layers' in config:
        print(f"  Finetune Early:      {config['finetune_early_layers']}")
    
    print(f"  Learning Rate:       {config.get('learning_rate', config.get('lr', 'N/A'))}")
    print(f"  Batch Size:          {config['batch_size']}")
    print(f"  Temperature:         {config['temperature']}")
    
    # Training results
    if 'epoch' in checkpoint:
        print(f"\n  Training Results:")
        print(f"    Epoch:             {checkpoint['epoch']}")
        print(f"    Best Val Loss:     {checkpoint.get('best_val_loss', 'N/A'):.6f}")
        if 'train_losses' in checkpoint and checkpoint['train_losses']:
            print(f"    Final Train Loss:  {checkpoint['train_losses'][-1]:.6f}")
        if 'val_losses' in checkpoint and checkpoint['val_losses']:
            print(f"    Final Val Loss:    {checkpoint['val_losses'][-1]:.6f}")

# Detailed comparison
print("\n" + "="*100)
print("DETAILED COMPARISON")
print("="*100)

# Model size comparison
print("\n🔍 Model Size Comparison:")
print("-" * 100)
model_params = {}
for name, checkpoint in checkpoints.items():
    params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
    model_params[name] = params
    print(f"  {name:35} {params:>15,} parameters  ({params/1e6:.2f}M)")

print(f"\n  Size Ratios:")
base_model = "Whisper Contrastive"
base_params = model_params[base_model]
for name, params in model_params.items():
    if name != base_model:
        ratio = params / base_params
        print(f"    {name} vs {base_model}: {ratio:.2f}x")

# Performance comparison
print("\n📈 Performance Comparison:")
print("-" * 100)
losses = {}
for name, checkpoint in checkpoints.items():
    if 'best_val_loss' in checkpoint:
        loss = checkpoint['best_val_loss']
        losses[name] = loss
        print(f"  {name:35} Best Val Loss: {loss:.6f}")

# Find winner
if losses:
    winner = min(losses, key=losses.get)
    best_loss = losses[winner]
    print(f"\n  🏆 Winner: {winner} with loss {best_loss:.6f}")
    
    print(f"\n  Performance Differences:")
    for model_name, loss in sorted(losses.items(), key=lambda x: x[1]):
        if model_name != winner:
            diff = loss - best_loss
            pct_diff = (diff / loss) * 100
            print(f"    {model_name:35} +{diff:.6f} ({pct_diff:.2f}% worse)")

# Architecture comparison
print("\n🏗️ Architecture Comparison:")
print("-" * 100)
print(f"  {'Model':<35} {'Type':<20} {'Proj Dim':<10} {'Features':<15} {'Finetune':<10}")
print("  " + "-" * 95)
for name in checkpoint_paths.keys():
    config = configs[name]
    model_type = "Contrastive" if "Contrastive" in name else "Adapter"
    proj_dim = config.get('projection_dim', 'N/A')
    features = f"{config.get('feature_dim', 0)}D acoustic" if 'feature_dim' in config else "None"
    finetune = "Yes" if config.get('finetune_early_layers') else "No"
    print(f"  {name:<35} {model_type:<20} {proj_dim!s:<10} {features:<15} {finetune:<10}")

# Training convergence
print("\n📉 Training Convergence:")
print("-" * 100)
for name, checkpoint in checkpoints.items():
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
        max_epochs = configs[name]['num_epochs']
        print(f"  {name:35} Stopped at epoch {epoch}/{max_epochs}")

# Recommendations
print("\n" + "="*100)
print("RECOMMENDATIONS")
print("="*100)

print("\n💡 Model Selection Guide:")

print("\n  🎯 Whisper Contrastive (Original):")
print("    ✓ Pure audio-based embeddings")
print("    ✓ Simpler architecture, no additional features")
print("    ✓ Good baseline for comparison")
print("    ✓ Larger projection dimension (128)")
if winner == "Whisper Contrastive":
    print("    ⭐ BEST VALIDATION PERFORMANCE!")

print("\n  🔧 Lightweight Adapter v1 (lr=0.001):")
print("    ✓ Adds 6D acoustic features (spectral/rhythm)")
print("    ✓ 2-layer transformer for feature integration")
print("    ✓ Smaller projection (32) + output (128)")
print("    ✓ No early layer finetuning")
if winner == "Lightweight Adapter (v1)":
    print("    ⭐ BEST VALIDATION PERFORMANCE!")

print("\n  🚀 Lightweight Adapter v2 (lr=0.0005, finetune):")
print("    ✓ Same acoustic features as v1")
print("    ✓ Early layer finetuning enabled")
print("    ✓ Lower learning rate (0.0005 vs 0.001)")
print("    ✓ Better adaptation of base model")
if winner == "Lightweight Adapter (v2)":
    print("    ⭐ BEST VALIDATION PERFORMANCE!")

print("\n  📊 Key Insights:")
print("    • Lightweight adapters add acoustic features without full model retraining")
print("    • v2 explores finetuning early layers with lower LR")
print("    • All models use the same Whisper base encoder")
print("    • Performance/size tradeoff varies by architecture")

print("\n" + "="*100)