# AI-Powered Garbage Classification System

## Overview
Deep learning application for classifying waste into multiple categories using CNNs and transfer learning. The project includes a training notebook and deployment-ready inference code.

Live Demo: Add your deployment link here.

## Features
- Real-time garbage classification
- Transfer learning with ResNet18
- Weighted loss to handle class imbalance
- Early stopping and learning rate scheduling
- Web interface for image upload and confidence scores

## Project Structure
```
.
├── Garbage_Classification.ipynb
├── model.py
├── app.py
├── app_gradio.py
├── templates/
│   └── index.html
├── Garbage_Classification_Outputs/
│   ├── best_model.pth
│   ├── classes.json
│   ├── confusion_matrix.png
│   └── training_metrics.png
├── requirements.txt
├── Dockerfile
└── README.md
```

## Dataset
- Source: Custom garbage dataset
- Classes: Derived from the dataset folders
- Split: 64% train, 16% val, 20% test

## Model Architecture
- Backbone: ResNet18 (ImageNet weights for initialization)
- Input size: 224x224
- Optimizer: Adam (lr=0.001)
- Scheduler: ReduceLROnPlateau
- Regularization: Dropout, batch normalization, weighted loss

## Installation
```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

## Usage

### Train in Notebook
Open the notebook and run all cells:
- Garbage_Classification.ipynb

### Run the Flask Web App
```bash
python app.py
```
Visit: http://localhost:5000

### Run the Gradio App
```bash
python app_gradio.py
```

### Run Inference from Python
```python
from model import GarbageClassifier

classifier = GarbageClassifier(
	"Garbage_Classification_Outputs/best_model.pth",
	classes_path="Garbage_Classification_Outputs/classes.json",
)
result = classifier.predict("trash.jpg")
print(result)
```

## Deployment Options
- Hugging Face Spaces (Gradio)
- Render.com (Docker + gunicorn)
- Any Docker-compatible platform

## Improvements Included
- Dropout and batch normalization
- Transfer learning option (ResNet18)
- Data augmentation with color jitter, affine transforms, and blur
- Class imbalance handling with weighted loss
- Early stopping and learning rate scheduling

## Future Enhancements
- Add webcam or video streaming inference
- Add model monitoring and logging
- Add model explainability tools (Grad-CAM)

## License
MIT

## Contact
Add your contact details here.
