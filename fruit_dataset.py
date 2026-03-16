import cv2
import os
import numpy as np
import pickle

def prepare_fruit_dataset():
    data = []
    labels = []
    fruit_info = {} # Mapping of label to (fruit_type, status)
    
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        print("Dataset directory not found.")
        return
        
    label_count = 0
    for fruit_dir in os.listdir(dataset_path):
        fruit_path = os.path.join(dataset_path, fruit_dir)
        if not os.path.isdir(fruit_path):
            continue
            
        try:
            fruit_type, status = fruit_dir.split('_')
        except ValueError:
            print(f"Skipping directory with incorrect format: {fruit_dir}")
            continue
            
        fruit_info[label_count] = (fruit_type, status)
        
        print(f"Processing images for {fruit_type} ({status})...")
        
        for img_name in os.listdir(fruit_path):
            img_path = os.path.join(fruit_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Resize and convert to HSV for better color-based feature extraction
            img_resized = cv2.resize(img, (64, 64))
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            
            # Use color histograms as features
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            
            features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            data.append(features)
            labels.append(label_count)
                
        label_count += 1
        
    if not data:
        print("No fruit data found in dataset.")
        return
        
    # Save processed data
    with open('processed_fruit_data.pkl', 'wb') as f:
        pickle.dump({'data': np.array(data), 'labels': np.array(labels), 'info': fruit_info}, f)
        
    print("Fruit dataset preparation complete. Saved to processed_fruit_data.pkl")

if __name__ == "__main__":
    prepare_fruit_dataset()
