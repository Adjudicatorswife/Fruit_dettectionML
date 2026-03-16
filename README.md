# Fruit Classification and Freshness Detection System

This Python-based system allows you to collect fruit image data, train a machine learning model, and perform real-time fruit classification and freshness detection using a live camera feed.

## Project Structure

- `collect_fruit_imgs.py`: Script to capture fruit images for different types and statuses.
- `fruit_dataset.py`: Script to extract color features and prepare the dataset.
- `train_fruit_model.py`: Script to train a Random Forest classifier.
- `fruit_inference.py`: Script for real-time fruit classification and freshness detection.
- `requirements.txt`: Lists all Python dependencies.
- `dataset/`: Directory where raw fruit images are stored.

## Prerequisites

### General Setup
Ensure you have Python 3.10.0 (or a compatible Python 3.x version) installed. It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Activate the virtual environment (macOS/Linux)
source venv/bin/activate

# Install required libraries from requirements.txt
pip install -r requirements.txt
```

### Windows Specific Notes

For Windows users, ensure you have Python added to your system's PATH during installation. If you encounter issues with `opencv-python`, you might try installing `opencv-contrib-python` instead. You can modify the `requirements.txt` file to `opencv-contrib-python` if needed.

This project uses `opencv`'s color histogram features and a Random Forest classifier for fruit classification and freshness detection. It does not directly use the `face-recognition` library which has complex dependencies like `dlib` and `cmake` that can be challenging to install on Windows. If you wish to explore more advanced image recognition libraries in the future, you might need to install CMake and Visual C++ Build Tools (available with Visual Studio).

## How to Use

### 1. Collect Fruit Data
Run the collection script for each fruit type and status you want to register. Replace `<fruit_type>` and `<status>` with the actual details.
```bash
python collect_fruit_imgs.py <fruit_type> <status>
```
Example:
```bash
python collect_fruit_imgs.py apple fresh
python collect_fruit_imgs.py apple rotten
python collect_fruit_imgs.py banana fresh
python collect_fruit_imgs.py banana rotten
```
The script will capture 100 images from your camera. Ensure good lighting and different angles. Press `q` to stop early.

### 2. Prepare the Dataset
After collecting data for all fruits, run the dataset preparation script to extract color features:
```bash
python fruit_dataset.py
```
This will create a `processed_fruit_data.pkl` file in the project root.

### 3. Train the Model
Train the machine learning model using the processed data:
```bash
python train_fruit_model.py
```
This will create a `trained_fruit_model.pkl` file in the project root.

### 4. Run Fruit Detection
Start the real-time inference script:
```bash
python fruit_inference.py
```
- **Fresh Fruits**: A green text will appear with the fruit type, "Fresh" status, and freshness percentage.
- **Rotten Fruits**: A red text will appear with the fruit type, "Rotten" status, and freshness percentage.
- **Unknown Objects**: A white bounding box and the label "unknown" will appear.
- **Bounding Boxes**: A colored bounding box will now appear around the detected fruit (Green for Fresh, Red for Rotten).

## Features
- **Real-time Detection**: Uses color histogram features for fast fruit classification.
- **Machine Learning**: Uses a Random Forest classifier for fruit type and status detection.
- **Freshness Estimation**: Calculates freshness percentage based on model confidence.
- **Visual Feedback**: Displays fruit type, status, and freshness percentage directly on the camera feed.
