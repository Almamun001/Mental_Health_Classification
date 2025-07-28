#!/usr/bin/env python3
"""
Mental Health Classification with PyTorch
Enhanced MentalBERT + BiLSTM + CNN Architecture

Author: AI Assistant
Objective: Achieve >93% validation accuracy with PyTorch-only implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

# Transformers and tokenization
from transformers import (
    AutoTokenizer, AutoModel, 
    get_linear_schedule_with_warmup,
    logging
)

# Data processing and analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings
from tqdm.auto import tqdm
import psutil

# Scikit-learn for metrics and preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

# Utility imports
import time
import random
from collections import Counter
import json
from pathlib import Path

# Configuration and styling
warnings.filterwarnings('ignore')
logging.set_verbosity_error()

class Config:
    """Configuration class for the mental health classification model"""
    
    # Hardware configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = torch.cuda.is_available()  # Mixed precision training
    
    # Model configuration
    MODEL_NAME = 'bert-base-uncased'  # Fallback to BERT if MentalBERT unavailable
    MAX_LENGTH = 512
    HIDDEN_SIZE = 768
    NUM_CLASSES = 7
    
    # Training configuration  
    BATCH_SIZE = 16 if torch.cuda.is_available() else 8
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 10
    WARMUP_STEPS = 500
    
    # Regularization
    DROPOUT = 0.3
    LSTM_DROPOUT = 0.2
    CNN_DROPOUT = 0.25
    LABEL_SMOOTHING = 0.1
    
    # Data configuration
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    TEST_SIZE = 0.1
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # File paths
    DATA_PATH = 'Mental health dataset (Final).csv'
    MODEL_SAVE_PATH = 'best_mental_health_model.pth'
    
    # HuggingFace token
    HF_TOKEN = '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'

def set_seed(seed):
    """Set random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_and_explore_data(data_path):
    """Load the dataset and perform initial exploration"""
    print("📊 Loading and exploring dataset...")
    
    # Load dataset
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Basic info
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"Missing values: {df.isnull().sum().to_dict()}")
    
    # Remove missing values
    initial_size = len(df)
    df = df.dropna(subset=['statement']).reset_index(drop=True)
    print(f"Removed {initial_size - len(df)} rows with missing statements")
    
    # Class distribution
    print("\n📈 Class Distribution:")
    class_counts = df['status'].value_counts()
    class_percentages = df['status'].value_counts(normalize=True) * 100
    
    for label, count in class_counts.items():
        percentage = class_percentages[label]
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    # Text statistics
    df['text_length'] = df['statement'].astype(str).str.len()
    df['word_count'] = df['statement'].astype(str).str.split().str.len()
    
    print("\n📝 Text Statistics:")
    print(f"  Average text length: {df['text_length'].mean():.1f} characters")
    print(f"  Average word count: {df['word_count'].mean():.1f} words")
    print(f"  Max text length: {df['text_length'].max():,} characters")
    print(f"  Min text length: {df['text_length'].min()} characters")
    
    return df

def advanced_text_preprocessing(text):
    """Apply advanced text preprocessing with careful cleaning"""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags and entities
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    
    # Remove URLs and links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'linkwww\S+', '', text)  # Handle malformed links
    
    # Clean up excessive whitespace while preserving sentence structure
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Handle repeated punctuation (keep some emotional emphasis)
    text = re.sub(r'([.!?]){3,}', r'\1\1\1', text)  # Max 3 repetitions
    text = re.sub(r'([,;:]){2,}', r'\1', text)  # Single for these
    
    return text

def preprocess_dataset(df):
    """Preprocess the entire dataset"""
    print("🧹 Preprocessing dataset...")
    
    # Apply text preprocessing
    tqdm.pandas(desc="Cleaning text")
    df['cleaned_statement'] = df['statement'].progress_apply(advanced_text_preprocessing)
    
    # Remove empty statements after cleaning
    before_cleaning = len(df)
    df = df[df['cleaned_statement'].str.len() > 0].reset_index(drop=True)
    after_cleaning = len(df)
    print(f"Removed {before_cleaning - after_cleaning} empty statements after cleaning")
    
    # Filter out extremely short or long texts that might be noise
    df['clean_length'] = df['cleaned_statement'].str.len()
    df['clean_word_count'] = df['cleaned_statement'].str.split().str.len()
    
    # Remove texts that are too short (likely noise) or too long (might be outliers)
    before_filter = len(df)
    df = df[
        (df['clean_word_count'] >= 3) &  # At least 3 words
        (df['clean_word_count'] <= 1000) &  # Max 1000 words
        (df['clean_length'] >= 10)  # At least 10 characters
    ].reset_index(drop=True)
    after_filter = len(df)
    print(f"Filtered out {before_filter - after_filter} texts (too short/long)")
    
    print(f"Final dataset size: {len(df):,} samples")
    
    return df

def prepare_data_splits(df):
    """Create stratified train/validation/test splits"""
    print("🔀 Creating stratified data splits...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['status'])
    
    # Create class mapping
    class_names = label_encoder.classes_
    label_to_name = {i: name for i, name in enumerate(class_names)}
    name_to_label = {name: i for i, name in enumerate(class_names)}
    
    print(f"Class mapping: {label_to_name}")
    
    # Stratified split: Train (80%), Temp (20%)
    X = df['cleaned_statement'].values
    y = df['label'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=Config.RANDOM_SEED
    )
    
    # Split temp into validation (10%) and test (10%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=Config.RANDOM_SEED
    )
    
    print(f"\n📊 Split sizes:")
    print(f"  Training: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'label_encoder': label_encoder,
        'class_names': class_names,
        'label_to_name': label_to_name,
        'name_to_label': name_to_label
    }

class MentalHealthDataset(Dataset):
    """Custom PyTorch Dataset for mental health text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class MentalHealthClassifier(nn.Module):
    """Enhanced Mental Health Classifier with MentalBERT + BiLSTM + CNN"""
    
    def __init__(self, model_name, num_classes, hidden_size=768, dropout=0.3):
        super(MentalHealthClassifier, self).__init__()
        
        # BERT backbone
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze lower layers to prevent overfitting
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[:6].parameters():  # Freeze first 6 layers
            param.requires_grad = False
        
        # BiLSTM layer for sequential modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size//2,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # CNN layers for local feature extraction
        self.conv1 = nn.Conv1d(hidden_size, hidden_size//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size//2, hidden_size//4, kernel_size=3, padding=1)
        
        # Global pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.lstm_dropout = nn.Dropout(0.2)
        self.cnn_dropout = nn.Dropout(0.25)
        
        # Classification layers
        combined_size = hidden_size + hidden_size//2  # BERT + LSTM + CNN features
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Dropout(dropout//2),
            nn.Linear(hidden_size//4, num_classes)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, input_ids, attention_mask):
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = bert_output.pooler_output  # [batch_size, hidden_size]
        
        # Apply layer normalization
        sequence_output = self.layer_norm(sequence_output)
        
        # BiLSTM processing
        lstm_output, _ = self.lstm(sequence_output)  # [batch_size, seq_len, hidden_size]
        lstm_output = self.lstm_dropout(lstm_output)
        
        # Global max pooling for LSTM features
        lstm_pooled = torch.max(lstm_output, dim=1)[0]  # [batch_size, hidden_size]
        
        # CNN processing
        cnn_input = sequence_output.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        cnn_output = F.relu(self.conv1(cnn_input))
        cnn_output = self.cnn_dropout(cnn_output)
        cnn_output = F.relu(self.conv2(cnn_output))
        
        # Global pooling for CNN features
        cnn_max_pooled = self.global_max_pool(cnn_output).squeeze(-1)  # [batch_size, hidden_size//4]
        cnn_avg_pooled = self.global_avg_pool(cnn_output).squeeze(-1)  # [batch_size, hidden_size//4]
        cnn_pooled = torch.cat([cnn_max_pooled, cnn_avg_pooled], dim=1)  # [batch_size, hidden_size//2]
        
        # Combine all features
        combined_features = torch.cat([pooled_output, lstm_pooled], dim=1)
        combined_features = self.dropout(combined_features)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits

def create_datasets_and_loaders(data_splits, config):
    """Create PyTorch datasets and data loaders"""
    print("🔗 Creating datasets and data loaders...")
    
    # Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        print(f"✅ Loaded tokenizer: {config.MODEL_NAME}")
    except Exception as e:
        print(f"⚠️ Failed to load {config.MODEL_NAME}: {e}")
        print("Falling back to bert-base-uncased")
        config.MODEL_NAME = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create datasets
    train_dataset = MentalHealthDataset(
        data_splits['X_train'], 
        data_splits['y_train'], 
        tokenizer, 
        config.MAX_LENGTH
    )
    
    val_dataset = MentalHealthDataset(
        data_splits['X_val'], 
        data_splits['y_val'], 
        tokenizer, 
        config.MAX_LENGTH
    )
    
    test_dataset = MentalHealthDataset(
        data_splits['X_test'], 
        data_splits['y_test'], 
        tokenizer, 
        config.MAX_LENGTH
    )
    
    # Calculate class weights for weighted sampling
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(data_splits['y_train']),
        y=data_splits['y_train']
    )
    class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)
    
    # Create weighted sampler for training
    sample_weights = [class_weights[label].item() for label in data_splits['y_train']]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"📦 Created datasets:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader, 
        'test_loader': test_loader,
        'tokenizer': tokenizer,
        'class_weights': class_weights
    }

def train_epoch(model, data_loader, optimizer, scheduler, scaler, device, class_weights):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Loss function with label smoothing and class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Mixed precision training
        with autocast(enabled=scaler is not None):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        
        # Backward pass with gradient scaling
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{total_correct/total_samples:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / len(data_loader), total_correct / total_samples

def evaluate_model(model, data_loader, device, class_weights):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            predictions = torch.argmax(logits, dim=1)
            
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy, all_predictions, all_labels

def train_model(model, train_loader, val_loader, config, class_weights):
    """Train the complete model"""
    print("🚀 Starting model training...")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP else None
    
    # Training tracking
    best_val_accuracy = 0
    training_stats = []
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n📅 Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Training
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, config.DEVICE, class_weights
        )
        
        # Validation
        val_loss, val_accuracy, val_predictions, val_labels = evaluate_model(
            model, val_loader, config.DEVICE, class_weights
        )
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'train_accuracy': train_accuracy
            }, config.MODEL_SAVE_PATH)
            print(f"✅ New best model saved! Validation accuracy: {val_accuracy:.4f}")
        
        # Log statistics
        stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
        training_stats.append(stats)
        
        print(f"📊 Epoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Early stopping check
        if val_accuracy > 0.93:
            print(f"🎯 Target validation accuracy (>93%) achieved: {val_accuracy:.4f}")
            break
    
    print(f"\n🏆 Best validation accuracy: {best_val_accuracy:.4f}")
    return training_stats, best_val_accuracy

def main():
    """Main function to run the mental health classification pipeline"""
    
    # Initialize configuration and set seed
    config = Config()
    set_seed(config.RANDOM_SEED)
    
    print("🖥️  System Configuration:")
    print(f"  Device: {config.DEVICE}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Mixed precision: {config.USE_AMP}")
    
    # Load and preprocess data
    df = load_and_explore_data(config.DATA_PATH)
    df = preprocess_dataset(df)
    
    # Create data splits
    data_splits = prepare_data_splits(df)
    config.NUM_CLASSES = len(data_splits['class_names'])
    
    # Create datasets and loaders
    data_loaders = create_datasets_and_loaders(data_splits, config)
    
    # Initialize model
    print(f"🧠 Initializing model: {config.MODEL_NAME}")
    model = MentalHealthClassifier(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train the model
    training_stats, best_val_accuracy = train_model(
        model, 
        data_loaders['train_loader'], 
        data_loaders['val_loader'], 
        config,
        data_loaders['class_weights']
    )
    
    # Load best model and evaluate on test set
    print("\n🔍 Loading best model for final evaluation...")
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_accuracy, test_predictions, test_labels = evaluate_model(
        model, data_loaders['test_loader'], config.DEVICE, data_loaders['class_weights']
    )
    
    print(f"\n🎯 Final Results:")
    print(f"  Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    report = classification_report(
        test_labels, 
        test_predictions, 
        target_names=data_splits['class_names'],
        digits=4
    )
    print(report)
    
    # Save training statistics
    import json
    with open('training_stats.json', 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    print("\n✅ Training completed successfully!")
    print(f"📁 Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"📈 Training stats saved to: training_stats.json")

if __name__ == "__main__":
    main()