"""
Train Deep Forest Model
Loads processed CodeComplex data and trains the model
"""

import pickle
from deep_forest_model import DeepForestModel

def train_model():
    """Train the Deep Forest model with processed dataset"""
    
    print("="*60)
    print("TRAINING DEEP FOREST MODEL")
    print("="*60)
    
    # Load processed dataset
    print("\nLoading processed dataset...")
    try:
        with open('data/balanced_data.pkl', 'rb') as f:
            training_data = pickle.load(f)
        print(f"✓ Loaded {len(training_data)} samples")
    except FileNotFoundError:
        print("❌ ERROR: processed_data.pkl not found!")
        print("\nPlease run 'python process_codecomplex.py' first")
        print("This will process your codecomplex.jsonl file")
        return
    
    # Initialize model
    print("\nInitializing Deep Forest model...")
    model = DeepForestModel()
    
    # Train model
    print("\nStarting training...")
    print("-"*60)
    
    result = model.train(training_data)
    
    print("-"*60)
    
    if result.get('success'):
        print("\n✅ TRAINING SUCCESSFUL!")
        print(f"\nModel Performance:")
        print(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"  Samples: {result['n_samples']}")
        print(f"  Features: {result['n_features']}")
        
        print(f"\n✓ Model saved to: models/trained_model.pkl")
        
        if result['accuracy'] >= 0.85:
            print(f"\n🎉 Target accuracy (85%) achieved!")
        else:
            print(f"\n⚠️  Accuracy below 85% target")
            print(f"   Consider:")
            print(f"   - Adding more training samples")
            print(f"   - Balancing complexity class distribution")
            print(f"   - Adjusting model parameters")
        
        print("\n" + "="*60)
        print("NEXT STEP: Run 'python app.py' to start the backend server")
        print("="*60)
    
    else:
        print("\n❌ Training failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == '__main__':
    train_model()