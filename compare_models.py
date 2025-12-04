import torch

# Load both checkpoints
checkpoint1_path = "/home/ar/Data/Ajitzi/deep-audio-embedding-visualization/ML/checkpoints/whisper_contrastive_20251128_125913/best_model.pth"
checkpoint2_path = "/home/ar/Data/Ajitzi/deep-audio-embedding-visualization/ML/checkpoints/whisper_contrastive_20251128_085448/best_model.pth"
checkpoint3_path = "/home/ar/Data/Ajitzi/deep-audio-embedding-visualization/ML/checkpoints/best_model.pth"

print("="*80)
print("MODEL COMPARISON")
print("="*80)

# Model 1 (Tiny)
print("\n📊 Model 1: whisper_contrastive_20251128_125913")
print("-" * 80)
checkpoint1 = torch.load(checkpoint1_path, map_location='cpu')
print(f"  Model Type:          Whisper-{checkpoint1['model_config']['model_name']}")
print(f"  Projection Dim:      {checkpoint1['model_config']['projection_dim']}")
print(f"  Epoch:               {checkpoint1['epoch']}")
print(f"  Best Val Loss:       {checkpoint1['best_val_loss']:.6f}")
print(f"  Final Train Loss:    {checkpoint1['train_losses'][-1]:.6f}")
print(f"  Final Val Loss:      {checkpoint1['val_losses'][-1]:.6f}")

# Model 2 (Base)
print("\n📊 Model 2: whisper_contrastive_20251128_085448")
print("-" * 80)
checkpoint2 = torch.load(checkpoint2_path, map_location='cpu')
print(f"  Model Type:          Whisper-{checkpoint2['model_config']['model_name']}")
print(f"  Projection Dim:      {checkpoint2['model_config']['projection_dim']}")
print(f"  Epoch:               {checkpoint2['epoch']}")
print(f"  Best Val Loss:       {checkpoint2['best_val_loss']:.6f}")
print(f"  Final Train Loss:    {checkpoint2['train_losses'][-1]:.6f}")
print(f"  Final Val Loss:      {checkpoint2['val_losses'][-1]:.6f}")

# Model 3 (First)
print("\n📊 Model 3: best_model.pth")
print("-" * 80)
checkpoint3 = torch.load(checkpoint3_path, map_location='cpu')
print(f"  Model Type:          Whisper-{checkpoint3['model_config']['model_name']}")
print(f"  Projection Dim:      {checkpoint3['model_config']['projection_dim']}")
print(f"  Epoch:               {checkpoint3['epoch']}")
print(f"  Best Val Loss:       {checkpoint3['best_val_loss']:.6f}")
print(f"  Final Train Loss:    {checkpoint3['train_losses'][-1]:.6f}")
print(f"  Final Val Loss:      {checkpoint3['val_losses'][-1]:.6f}")

# Comparison
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

# Model size comparison
print("\n🔍 Model Size:")
model1_params = sum(p.numel() for p in checkpoint1['model_state_dict'].values())
model2_params = sum(p.numel() for p in checkpoint2['model_state_dict'].values())
model3_params = sum(p.numel() for p in checkpoint3['model_state_dict'].values())
print(f"  Model 1 (Tiny):  {model1_params:,} parameters")
print(f"  Model 2 (Base):  {model2_params:,} parameters")
print(f"  Model 3 (First): {model3_params:,} parameters")
print(f"\n  Size ratios:")
print(f"    Model 2 vs Model 1: {model2_params/model1_params:.2f}x")
print(f"    Model 3 vs Model 1: {model3_params/model1_params:.2f}x")

# Performance comparison
print("\n📈 Performance:")
losses = {
    'Model 1 (Tiny)': checkpoint1['best_val_loss'],
    'Model 2 (Base)': checkpoint2['best_val_loss'],
    'Model 3 (First)': checkpoint3['best_val_loss']
}

# Display all losses
for model_name, loss in losses.items():
    print(f"  {model_name}: {loss:.6f}")

# Find winner
winner = min(losses, key=losses.get)
best_loss = losses[winner]
print(f"\n  🏆 Winner: {winner} with loss {best_loss:.6f}")

# Show improvement over others
print(f"\n  Performance advantages:")
for model_name, loss in losses.items():
    if model_name != winner:
        diff = loss - best_loss
        pct_diff = (diff / loss) * 100
        print(f"    vs {model_name}: {diff:.6f} lower ({pct_diff:.2f}% better)")

# Training curve analysis
print("\n📉 Training Convergence:")
print(f"  Model 1 epochs to best: {checkpoint1['epoch']}")
print(f"  Model 2 epochs to best: {checkpoint2['epoch']}")
print(f"  Model 3 epochs to best: {checkpoint3['epoch']}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("\n💡 Choose based on your use case:")

print("\n  • Use Model 1 (Tiny) if:")
print("    - Speed/efficiency is critical")
print("    - Running on resource-constrained devices")
print("    - Real-time inference is needed")
print("    - Model size matters (deployment, storage)")
if winner == "Model 1 (Tiny)":
    print("    - ⭐ It has the BEST validation performance!")

print("\n  • Use Model 2 (Base) if:")
print("    - Best possible accuracy is needed")
print("    - Resources are available")
print("    - Richer feature representations are desired")
print("    - Fine-grained genre distinctions are important")
if winner == "Model 2 (Base)":
    print("    - ⭐ It has the BEST validation performance!")

print("\n  • Use Model 3 (First) if:")
print("    - You want the original/baseline model")
print("    - Comparing against initial training results")
if winner == "Model 3 (First)":
    print("    - ⭐ It has the BEST validation performance!")

print("\n" + "="*80)
