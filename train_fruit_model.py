import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_fruit_model():
    # Load processed fruit data
    try:
        with open('processed_fruit_data.pkl', 'rb') as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print("Processed fruit data not found. Run fruit_dataset.py first.")
        return
        
    X = dataset['data']
    y = dataset['labels']
    info = dataset['info']
    
    if len(X) == 0:
        print("No data to train on.")
        return
        
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Fruit model trained with accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained model and fruit info
    with open('trained_fruit_model.pkl', 'wb') as f:
        pickle.dump({'model': rf, 'info': info}, f)
        
    print("Trained fruit model saved to trained_fruit_model.pkl")

if __name__ == "__main__":
    train_fruit_model()
