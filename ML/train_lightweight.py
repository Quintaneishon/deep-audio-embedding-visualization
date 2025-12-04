"""
Training script for Lightweight Adapter.

Trains the LightweightAdapter model on MTG-Jamendo dataset using supervised
contrastive loss with acoustic features.

Adaptations from train_contrastive.py:
- Uses LightweightAdapter instead of WhisperContrastive
- Uses dataset_with_features for feature loading
- Supports mixed precision training (FP16) for memory efficiency
- Optimized hyperparameters for lightweight adapter
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ML.models.LightweightAdapter import LightweightAdapter
from ML.losses import SupConLoss
from ML.dataset_with_features import create_dataloaders_with_features
from config import PROJECT_ROOT


class LightweightAdapterTrainer:
    """Trainer class for Lightweight Adapter with feature loading."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        checkpoint_dir,
        log_dir,
        num_epochs=35,
        early_stopping_patience=10,
        mixed_precision=False
    ):
        """
        Initialize trainer.
        
        Args:
            model: LightweightAdapter model
            train_loader: Training data loader (returns audio, label, features)
            val_loader: Validation data loader
            criterion: Loss function (SupConLoss)
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            num_epochs: Number of training epochs
            early_stopping_patience: Epochs to wait before early stopping
            mixed_precision: Use mixed precision training (FP16)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.mixed_precision = mixed_precision
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
        print(f"\nTrainer initialized:")
        print(f"  Mixed precision: {mixed_precision}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        # Ensure only adapter layers are trainable
        for param in self.model.feature_projection.parameters():
            param.requires_grad = True
        for param in self.model.fusion.parameters():
            param.requires_grad = True
        
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs}')
        for batch_idx, (audio, labels, features) in enumerate(pbar):
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            features = features.to(self.device)
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    embeddings = self.model(audio, features)
                    loss = self.criterion(embeddings, labels)
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                embeddings = self.model(audio, features)
                loss = self.criterion(embeddings, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        epoch_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for audio, labels, features in tqdm(self.val_loader, desc='Validation'):
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                features = features.to(self.device)
                
                # Forward pass
                embeddings = self.model(audio, features)
                
                # Compute loss
                loss = self.criterion(embeddings, labels)
                
                epoch_loss += loss.item()
                num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """
        Save model checkpoint.
        
        Args:
            filename: Name of checkpoint file
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'model_name': self.model.model_name,
                'feature_dim': self.model.feature_dim,
                'projection_dim': self.model.projection_dim,
                'output_dim': self.model.output_dim,
                'n_audio_state': self.model.n_audio_state
            }
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
            
            # Also save feature normalization stats
            stats_path = self.checkpoint_dir / 'feature_stats.json'
            self.model.save_feature_stats(str(stats_path))
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Training - Lightweight Adapter")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("="*60 + "\n")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Log metrics
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            if self.scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6f}")
                self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
            
            # Save checkpoint every epoch
            self.save_checkpoint(filename=f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
                print(f"  ★ New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.early_stopping_patience}")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            print()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60 + "\n")
        
        self.writer.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Lightweight Adapter Model')
    
    # Dataset arguments
    parser.add_argument('--data-root', type=str,
                        default='/home/ar/Data/Ajitzi/mtg-jamendo-dataset',
                        help='Root directory of MTG-Jamendo dataset')
    parser.add_argument('--split', type=str, default='split-0',
                        help='Data split to use')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='base',
                        choices=['tiny', 'base', 'small'],
                        help='Whisper model size')
    parser.add_argument('--feature-dim', type=int, default=6,
                        help='Dimension of acoustic features')
    parser.add_argument('--projection-dim', type=int, default=32,
                        help='Dimension to project features to')
    parser.add_argument('--output-dim', type=int, default=128,
                        help='Output embedding dimension')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Batch size (optimized for GPU memory)')
    parser.add_argument('--num-epochs', type=int, default=35,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training (FP16)')
    
    # Feature arguments
    parser.add_argument('--feature-cache', type=str, default=None,
                        help='Path to HDF5 feature cache file')
    parser.add_argument('--feature-stats', type=str, default=None,
                        help='Path to feature normalization stats JSON')
    parser.add_argument('--extract-on-fly', action='store_true',
                        help='Extract features on-the-fly instead of using cache')
    
    # Data loading arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Audio duration in seconds')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Audio sample rate')
    parser.add_argument('--balanced-sampling', action='store_true',
                        help='Use balanced sampling for training')
    
    # Contrastive loss arguments
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory for TensorBoard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--cuda-number', type=str, default=None,
                        help='CUDA number to use')
    
    args = parser.parse_args()
    
    # Set default checkpoint and log directories
    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.checkpoint_dir = os.path.join(
            PROJECT_ROOT, 'ML', 'checkpoints', f'lightweight_adapter_{timestamp}'
        )
    
    if args.log_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_dir = os.path.join(
            PROJECT_ROOT, 'ML', 'logs', f'lightweight_adapter_{timestamp}'
        )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda_number is not None:
        device = torch.device(f'cuda:{args.cuda_number}')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device.index)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1e9:.2f} GB\n")
    
    # Prepare dataset paths
    data_root = Path(args.data_root)
    splits_dir = data_root / 'data' / 'splits' / args.split
    audio_dir = data_root / 'songs'
    
    train_tsv = splits_dir / 'autotagging_genre-train.tsv'
    val_tsv = splits_dir / 'autotagging_genre-validation.tsv'
    test_tsv = splits_dir / 'autotagging_genre-test.tsv'
    
    print(f"Dataset configuration:")
    print(f"  Data root: {data_root}")
    print(f"  Split: {args.split}")
    print(f"  Audio directory: {audio_dir}\n")
    
    # Create dataloaders with features
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, dataset_info = create_dataloaders_with_features(
        str(train_tsv),
        str(val_tsv),
        str(test_tsv),
        str(audio_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        duration=args.duration,
        balanced_sampling=args.balanced_sampling,
        feature_cache_file=args.feature_cache,
        feature_stats_file=args.feature_stats,
        extract_on_fly=args.extract_on_fly
    )
    
    print(f"\nDataset info:")
    print(f"  Num classes: {dataset_info['num_classes']}")
    print(f"  Train size: {dataset_info['train_size']}")
    print(f"  Val size: {dataset_info['val_size']}")
    print(f"  Test size: {dataset_info['test_size']}")
    print(f"  Feature dim: {dataset_info['feature_dim']}\n")
    
    # Create model
    print("Creating Lightweight Adapter model...")
    model = LightweightAdapter(
        model_name=args.model_name,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
        output_dim=args.output_dim,
        device=device,
        feature_stats_path=args.feature_stats
    )
    model.to(device)
    
    # Create loss function
    criterion = SupConLoss(temperature=args.temperature)
    
    # Create optimizer (only for trainable parameters)
    optimizer = optim.Adam(
        model.get_trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer = LightweightAdapterTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping,
        mixed_precision=args.mixed_precision
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Save training configuration
    config_path = Path(args.checkpoint_dir) / 'training_config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Training config saved to {config_path}\n")
    
    # Save dataset info
    dataset_info_path = Path(args.checkpoint_dir) / 'dataset_info.json'
    with open(dataset_info_path, 'w') as f:
        # Convert distribution Counter to dict for JSON serialization
        info_copy = dataset_info.copy()
        info_copy['train_distribution'] = dict(info_copy['train_distribution'])
        json.dump(info_copy, f, indent=2)
    print(f"Dataset info saved to {dataset_info_path}\n")
    
    # Start training
    trainer.train()
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()

