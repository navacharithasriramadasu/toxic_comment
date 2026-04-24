#!/usr/bin/env python3
"""
Create Sample Dataset for Toxic Comment Detection
Summer Internship Project | Summer of AI
"""

import pandas as pd
import os

def create_sample_dataset():
    """Create a sample dataset for training"""
    
    # Sample data with various toxicity levels
    data = [
        # Toxic comments
        {"id": 0, "comment_text": "You are so stupid!", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 1, "comment_text": "Go kill yourself you worthless piece of trash", "toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 1, "insult": 1, "identity_hate": 0},
        {"id": 2, "comment_text": "This is the worst thing I've ever seen", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 3, "comment_text": "I'm going to find you and hurt you", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        {"id": 4, "comment_text": "You're an idiot and nobody likes you", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 5, "comment_text": "Fuck you asshole", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 6, "comment_text": "Die in a fire you scum", "toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 1, "insult": 1, "identity_hate": 0},
        {"id": 7, "comment_text": "You're a disgusting human being", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 8, "comment_text": "I will track you down and kill your family", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        {"id": 9, "comment_text": "Stupid people shouldn't be allowed to post", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 10, "comment_text": "Your opinion is worthless", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 11, "comment_text": "Go back to where you came from", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 1},
        {"id": 12, "comment_text": "You're mentally retarded", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 13, "comment_text": "This is shit and you know it", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 14, "comment_text": "I hope you get cancer and die", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        {"id": 15, "comment_text": "You're a waste of oxygen", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 16, "comment_text": "Burn in hell you sinner", "toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 1, "insult": 0, "identity_hate": 0},
        {"id": 17, "comment_text": "This is pathetic", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 18, "comment_text": "You should be ashamed of yourself", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 19, "comment_text": "I'm reporting this to the authorities", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        {"id": 20, "comment_text": "You're a terrible person", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 21, "comment_text": "Go fuck yourself with a chainsaw", "toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 1, "insult": 1, "identity_hate": 0},
        
        # Non-toxic comments
        {"id": 22, "comment_text": "I love this product, it's amazing!", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 23, "comment_text": "Thank you for sharing this information", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 24, "comment_text": "Great work! Keep it up!", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 25, "comment_text": "Have a wonderful day everyone", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 26, "comment_text": "This is helpful and constructive feedback", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 27, "comment_text": "I appreciate your perspective on this topic", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 28, "comment_text": "Well written article with good points", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 29, "comment_text": "Excellent analysis and insights", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 30, "comment_text": "Thanks for the detailed explanation", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 31, "comment_text": "Outstanding research and presentation", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 32, "comment_text": "Interesting point, I hadn't considered that", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 33, "comment_text": "I learned something new today, thank you", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 34, "comment_text": "This changed my mind on the issue", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 35, "comment_text": "I respect your viewpoint even if I disagree", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 36, "comment_text": "Important contribution to the discussion", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 37, "comment_text": "I see your point and it makes sense", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 38, "comment_text": "Creative solution to a complex problem", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 39, "comment_text": "This deserves wider recognition", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 40, "comment_text": "Valuable insights, thank you for sharing", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 41, "comment_text": "This deserves an award", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 42, "comment_text": "I appreciate the effort you put in", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = "data/train.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Sample dataset created: {output_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📝 Total comments: {len(df)}")
    
    # Show statistics
    print("\n📈 Dataset Statistics:")
    toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for col in toxic_cols:
        count = df[col].sum()
        percentage = (count / len(df)) * 100
        print(f"  {col}: {count} ({percentage:.1f}%)")
    
    print(f"\n🎯 Balance: {df['toxic'].sum()} toxic, {len(df) - df['toxic'].sum()} non-toxic")
    
    return df

if __name__ == "__main__":
    create_sample_dataset()
