import cv2
import pickle
import numpy as np
import os

def run_fruit_inference():
    # Load trained fruit model and info
    try:
        with open('trained_fruit_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            rf = model_data['model']
            fruit_info = model_data['info']
    except FileNotFoundError:
        print("Trained fruit model not found. Run train_fruit_model.py first.")
        return
        
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    print("Starting fruit inference. Press 'q' to stop.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Process the frame for prediction
        img_resized = cv2.resize(frame, (64, 64))
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        
        # Extract color histogram features
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()]).reshape(1, -1)
        
        # Predict fruit type and status
        prediction = rf.predict(features)[0]
        probabilities = rf.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        # Define a bounding box (since this is a global classifier, we'll draw a box around the center)
        # In a real object detection system, this would come from the detector.
        # For this classification-based project, we'll simulate a central detection box.
        box_margin = 100
        x1, y1 = max(0, width//2 - box_margin), max(0, height//2 - box_margin)
        x2, y2 = min(width, width//2 + box_margin), min(height, height//2 + box_margin)
        
        # Threshold for unknown
        if confidence > 0.4:
            fruit_type, status = fruit_info[prediction]
            
            # Calculate freshness percentage
            if status == 'fresh':
                freshness_pct = int(confidence * 100)
                display_status = "Fresh"
                color = (0, 255, 0) # Green for fresh
            else:
                freshness_pct = int((1 - confidence) * 100)
                display_status = "Rotten"
                color = (0, 0, 255) # Red for rotten
                
            # Ensure freshness_pct is within 0-100
            freshness_pct = max(0, min(100, freshness_pct))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display info
            label = f"{fruit_type.capitalize()} ({display_status})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Freshness: {freshness_pct}%", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # Draw bounding box for unknown
            color = (255, 255, 255) # White for unknown
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "unknown", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        cv2.imshow('Fruit Freshness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_fruit_inference()
