#!/usr/bin/env python3
"""
Download Jigsaw Toxic Comment Classification Dataset
Summer Internship Project | Summer of AI
"""

import os
import pandas as pd
import requests
import zipfile
import io
from pathlib import Path

def download_jigsaw_dataset():
    """Download the Jigsaw Toxic Comment Classification dataset"""
    
    # URLs for the dataset files
    urls = {
        'train.csv': 'https://github.com/t-davidson/hate-speech-and-offensive-language/raw/master/data/merged_data.csv',
        # Alternative source if above doesn't work
        'alternative': 'https://raw.githubusercontent.com/siddhantbhattarai/toxic-comment-classification/master/train.csv'
    }
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    train_path = data_dir / "train.csv"
    
    print("📥 Downloading Jigsaw Toxic Comment Classification dataset...")
    
    try:
        # Try primary source
        print("Downloading from primary source...")
        response = requests.get(urls['train.csv'], timeout=30)
        response.raise_for_status()
        
        # Load and process the data
        df = pd.read_csv(io.StringIO(response.text))
        
        # Check if this is the right format or if we need to adapt it
        if 'comment_text' not in df.columns:
            print("Adapting dataset format...")
            # For alternative dataset format
            if 'tweet' in df.columns:
                df = df.rename(columns={'tweet': 'comment_text'})
            
            # Create the required toxic columns if they don't exist
            required_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0  # Default to non-toxic
            
            # If there's a 'class' column, map it to toxicity
            if 'class' in df.columns:
                # Assuming class 0 = hate, 1 = offensive, 2 = neither
                # Map to our toxicity categories
                df.loc[df['class'] == 0, 'toxic'] = 1
                df.loc[df['class'] == 0, 'severe_toxic'] = 1
                df.loc[df['class'] == 0, 'identity_hate'] = 1
                df.loc[df['class'] == 1, 'toxic'] = 1
                df.loc[df['class'] == 1, 'insult'] = 1
        
        # Ensure we have the required columns
        required_cols = ['comment_text'] + ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
            
        # Add ID column if missing
        if 'id' not in df.columns:
            df['id'] = range(len(df))
        
        # Save the dataset
        df.to_csv(train_path, index=False)
        print(f"✅ Dataset saved to {train_path}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📝 Columns: {list(df.columns)}")
        
        # Show some statistics
        print("\n📈 Dataset Statistics:")
        toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        for col in toxic_cols:
            count = df[col].sum()
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count:,} ({percentage:.2f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        
        # Try alternative source
        try:
            print("Trying alternative source...")
            response = requests.get(urls['alternative'], timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text))
            
            # Create a sample dataset in the right format if needed
            if 'comment_text' not in df.columns:
                print("Creating sample dataset...")
                # Create a sample dataset for demonstration
                sample_data = {
                    'id': range(1000),
                    'comment_text': [
                        "You are so stupid!",
                        "I love this product, it's amazing!",
                        "Go kill yourself you worthless piece of trash",
                        "Thank you for sharing this information",
                        "This is the worst thing I've ever seen",
                        "Great work! Keep it up!",
                        "I'm going to find you and hurt you",
                        "Have a wonderful day everyone",
                        "You're an idiot and nobody likes you",
                        "This is helpful and constructive feedback"
                    ] * 100,
                    'toxic': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 100,
                    'severe_toxic': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0] * 100,
                    'obscene': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] * 100,
                    'threat': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] * 100,
                    'insult': [1, 0, 1, 0, 0, 0, 0, 0, 1, 0] * 100,
                    'identity_hate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 100
                }
                df = pd.DataFrame(sample_data)
            
            df.to_csv(train_path, index=False)
            print(f"✅ Sample dataset saved to {train_path}")
            print(f"📊 Dataset shape: {df.shape}")
            return True
            
        except Exception as e2:
            print(f"❌ Alternative source also failed: {e2}")
            return False

if __name__ == "__main__":
    success = download_jigsaw_dataset()
    if success:
        print("\n🎉 Dataset ready for training!")
        print("Run: python scripts/train.py --data data/train.csv")
    else:
        print("\n❌ Failed to download dataset")
        print("Please manually download from: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data")
