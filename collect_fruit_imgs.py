import cv2
import os
import sys

def collect_fruit_images(fruit_type, status):
    # Create directory for the fruit type and status (e.g., apple_fresh)
    directory = f"dataset/{fruit_type}_{status}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    count = 0
    print(f"Starting collection for {fruit_type} ({status}). Press 'q' to stop or wait for 100 images.")
    
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display the frame
        cv2.putText(frame, f"Collecting: {count}/100", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Collecting Fruit Images', frame)
        
        # Save image
        img_path = os.path.join(directory, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Successfully collected {count} images for {fruit_type} ({status}).")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python collect_fruit_imgs.py <fruit_type> <status>")
        print("Example: python collect_fruit_imgs.py apple fresh")
    else:
        fruit = sys.argv[1]
        status = sys.argv[2]
        collect_fruit_images(fruit, status)
