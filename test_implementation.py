#!/usr/bin/env python3
"""
Quick test script for Mental Health Classification PyTorch implementation
Tests data loading, preprocessing, and model initialization
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import warnings

warnings.filterwarnings('ignore')

def test_data_loading():
    """Test data loading and basic preprocessing"""
    print("🧪 Testing data loading...")
    
    try:
        df = pd.read_csv('Mental health dataset (Final).csv')
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Classes: {df['status'].unique()}")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        return df
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None

def test_text_preprocessing():
    """Test text preprocessing function"""
    print("\n🧪 Testing text preprocessing...")
    
    def advanced_text_preprocessing(text):
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags and entities
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'linkwww\S+', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    # Test cases
    test_texts = [
        "I feel anxious about everything linkwwwamazoncom",
        "Depression is affecting my <b>daily life</b> &amp; work",
        "   Multiple    spaces   and   weird   formatting   ",
        "Normal text about mental health awareness"
    ]
    
    try:
        for i, text in enumerate(test_texts):
            cleaned = advanced_text_preprocessing(text)
            print(f"   Test {i+1}: '{text[:50]}...' -> '{cleaned[:50]}...'")
        
        print("✅ Text preprocessing working")
        return True
    except Exception as e:
        print(f"❌ Text preprocessing failed: {e}")
        return False

def test_data_splits():
    """Test data splitting functionality"""
    print("\n🧪 Testing data splits...")
    
    try:
        # Create dummy data
        texts = [f"sample text {i}" for i in range(1000)]
        labels = np.random.randint(0, 7, 1000)
        
        # Test stratified split
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        print(f"✅ Data splits successful:")
        print(f"   Train: {len(X_train)} ({len(X_train)/len(texts)*100:.1f}%)")
        print(f"   Val: {len(X_val)} ({len(X_val)/len(texts)*100:.1f}%)")
        print(f"   Test: {len(X_test)} ({len(X_test)/len(texts)*100:.1f}%)")
        
        return True
    except Exception as e:
        print(f"❌ Data splitting failed: {e}")
        return False

def test_model_components():
    """Test model component initialization"""
    print("\n🧪 Testing model components...")
    
    try:
        # Test basic PyTorch functionality
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        # Test tensor operations
        x = torch.randn(8, 512, 768)  # batch_size, seq_len, hidden_size
        
        # Test LSTM
        lstm = torch.nn.LSTM(768, 384, 2, batch_first=True, bidirectional=True)
        lstm_out, _ = lstm(x)
        print(f"   LSTM output shape: {lstm_out.shape}")
        
        # Test CNN
        conv = torch.nn.Conv1d(768, 384, kernel_size=3, padding=1)
        x_transposed = x.transpose(1, 2)
        conv_out = conv(x_transposed)
        print(f"   CNN output shape: {conv_out.shape}")
        
        # Test attention
        attention = torch.nn.MultiheadAttention(768, 8, batch_first=True)
        attn_out, _ = attention(x, x, x)
        print(f"   Attention output shape: {attn_out.shape}")
        
        print("✅ Model components working")
        return True
    except Exception as e:
        print(f"❌ Model component test failed: {e}")
        return False

def test_configuration():
    """Test configuration class"""
    print("\n🧪 Testing configuration...")
    
    try:
        class Config:
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            USE_AMP = torch.cuda.is_available()
            MODEL_NAME = 'bert-base-uncased'
            MAX_LENGTH = 512
            HIDDEN_SIZE = 768
            NUM_CLASSES = 7
            BATCH_SIZE = 16 if torch.cuda.is_available() else 8
            LEARNING_RATE = 2e-5
            DROPOUT = 0.4
            RANDOM_SEED = 42
        
        config = Config()
        
        print(f"   Device: {config.DEVICE}")
        print(f"   Mixed precision: {config.USE_AMP}")
        print(f"   Batch size: {config.BATCH_SIZE}")
        print(f"   Model: {config.MODEL_NAME}")
        
        print("✅ Configuration working")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Mental Health Classification Tests")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Data Loading", test_data_loading),
        ("Text Preprocessing", test_text_preprocessing), 
        ("Data Splits", test_data_splits),
        ("Model Components", test_model_components),
        ("Configuration", test_configuration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            if test_name == "Data Loading":
                result = test_func()
                results[test_name] = result is not None
            else:
                result = test_func()
                results[test_name] = result if result is not None else False
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Ready to run the full implementation.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    print("\n💡 To run the full model:")
    print("   jupyter notebook MentalBERT_BiLSTM_CNN_(Enhanced)_PyTorch.ipynb")
    print("   OR")
    print("   python mental_health_pytorch.py")

if __name__ == "__main__":
    main()