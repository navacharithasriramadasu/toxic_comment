#!/usr/bin/env python3
"""
Generate Negative Comments for Testing
Summer Internship Project | Summer of AI
"""

import pandas as pd

def generate_negative_comments():
    """Generate various types of negative comments for testing"""
    
    # Different categories of negative comments
    comments = [
        # General negative/toxic
        {"id": 1, "comment_text": "You are so stupid!", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 2, "comment_text": "This is the worst thing I've ever seen", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 3, "comment_text": "I hate this product so much", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 4, "comment_text": "Completely useless and terrible", "tovere": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 5, "comment_text": "What a waste of time and money", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 6, "comment_text": "You're an idiot who doesn't know anything", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 7, "comment_text": "This is absolute garbage", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 8, "comment_text": "I can't believe how bad this is", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 9, "comment_text": "Totally disappointed with this", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 10, "comment_text": "This makes me angry just looking at it", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        
        # Severe toxic + threats
        {"id": 11, "comment_text": "Go kill yourself you worthless piece of trash", "toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 1, "insult": 1, "identity_hate": 0},
        {"id": 12, "comment_text": "Die in a fire you scum", "toxic": 1, "severe_toxic": 1, "obscene": 1, "threat": 1, "insult": 1, "identity_hate": 0},
        {"id": 13, "comment_text": "I hope you get cancer and die", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        {"id": 14, "comment_text": "I'm going to find you and hurt you", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        {"id": 15, "comment_text": "I will track you down and kill your family", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 0, "identity_hate": 0},
        
        # Obscene language
        {"id": 16, "comment_text": "Fuck you asshole", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 17, "comment_text": "This is shit and you know it", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 18, "comment_text": "What the hell is this crap", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 19, "comment_text": "Damn this stupid thing to hell", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 20, "comment_text": "Bullshit, this is completely fake", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 0, "identity_hate": 0},
        
        # Insults
        {"id": 21, "comment_text": "You're a disgusting human being", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 22, "comment_text": "You should be ashamed of yourself", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 23, "comment_text": "You're a terrible person", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 24, "comment_text": "Nobody cares about your opinion", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        {"id": 25, "comment_text": "You're a waste of oxygen", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
        
        # Identity hate
        {"id": 26, "comment_text": "Go back to where you came from", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 1},
        {"id": 27, "comment_text": "People like you don't belong here", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 1},
        {"id": 28, "comment_text": "Your kind ruins everything", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 1},
        {"id": 29, "comment_text": "Stay in your own country", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 1},
        {"id": 30, "comment_text": "You're a disgrace to your race", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 1},
        
        # Mild negative (for comparison)
        {"id": 31, "comment_text": "This is absolutely terrible", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 32, "comment_text": "I hate everything about this", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 33, "comment_text": "This is completely unacceptable", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 34, "comment_text": "I'm so disappointed", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        {"id": 35, "comment_text": "This fails on every level", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
    ]
    
    # Create DataFrame
    df = pd.DataFrame(comments)
    
    # Save to CSV
    output_path = "data/test_negative_comments.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Negative comments dataset created: {output_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📝 Total comments: {len(df)}")
    
    # Show statistics
    print("\n📈 Dataset Statistics:")
    toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for col in toxic_cols:
        count = df[col].sum()
        percentage = (count / len(df)) * 100
        print(f"  {col}: {count} ({percentage:.1f}%)")
    
    print(f"\n🎯 Categories:")
    print(f"  General toxic: {len(df)} comments")
    print(f"  Severe toxic: {df['severe_toxic'].sum()} comments")
    print(f"  Obscene: {df['obscene'].sum()} comments")
    print(f"  Threats: {df['threat'].sum()} comments")
    print(f"  Insults: {df['insult'].sum()} comments")
    print(f"  Identity hate: {df['identity_hate'].sum()} comments")
    
    # Show sample comments by category
    print(f"\n📝 Sample Comments:")
    print(f"\n🔴 Severe Toxic/Threats:")
    severe_comments = df[df['severe_toxic'] == 1]['comment_text'].head(3).tolist()
    for comment in severe_comments:
        print(f"  - {comment}")
    
    print(f"\n🟠 Obscene Language:")
    obscene_comments = df[df['obscene'] == 1]['comment_text'].head(3).tolist()
    for comment in obscene_comments:
        print(f"  - {comment}")
    
    print(f"\n🟡 Insults:")
    insult_comments = df[df['insult'] == 1]['comment_text'].head(3).tolist()
    for comment in insult_comments:
        print(f"  - {comment}")
    
    print(f"\n🟣 Identity Hate:")
    hate_comments = df[df['identity_hate'] == 1]['comment_text'].head(3).tolist()
    for comment in hate_comments:
        print(f"  - {comment}")
    
    return df

if __name__ == "__main__":
    generate_negative_comments()
