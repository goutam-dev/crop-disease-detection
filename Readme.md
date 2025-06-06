# Smart Crop Disease Detection System

A deep learning-powered web application for automatic plant disease detection using Convolutional Neural Networks (CNN). This system can identify diseases in plant leaves from uploaded images with high accuracy across 38 different disease categories.

## 🌟 Features

- **Automated Disease Detection**: Upload plant leaf images and get instant disease predictions
- **Multi-Class Classification**: Supports 38 different plant disease categories
- **User-Friendly Interface**: Clean, intuitive Streamlit web interface
- **High Accuracy**: CNN model trained on 87K+ images with robust performance
- **Real-Time Validation**: Instant feedback on prediction accuracy for test images

## 🎯 Project Overview
The system uses advanced deep learning techniques to analyze plant leaf images and detect various diseases, helping farmers and agricultural professionals make quick, informed decisions about crop health.

## 🛠️ Technology Stack

### Programming Language & Frameworks
- **Python**: Core programming language
- **TensorFlow & Keras**: Deep learning model development
- **Streamlit**: Web application framework

### Libraries & Dependencies
- **NumPy**: Numerical computing and array operations
- **OpenCV (cv2)**: Image preprocessing and computer vision
- **Matplotlib**: Data visualization and plotting
- **Seaborn**: Statistical data visualization
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Model evaluation metrics
- **KaggleHub**: Dataset management
- **Glob & OS**: File system operations

### Model Architecture
- **CNN (Convolutional Neural Network)**: Primary classification model
- **Input Size**: 128×128 RGB images
- **Classes**: 38 plant disease categories
- **Data Augmentation**: Random flip, rotation, and zoom

## 📊 Dataset

**Source**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/) from Kaggle

### Dataset Statistics
- **Total Images**: ~87,000 RGB images
- **Training Set**: 70,295 images (80%)
- **Validation Set**: 17,572 images (20%)
- **Test Set**: 33 images for prediction testing
- **Classes**: 38 different plant disease categories
- **Image Quality**: High-resolution, labeled images
- **Balance**: Well-balanced dataset across all classes


## 🏗️ Model Architecture

### CNN Architecture Details

**Input Layer**
- Input shape: 128×128×3 (RGB images)
- Data augmentation: RandomFlip, RandomRotation, RandomZoom

**Convolutional Feature Extractor**
- 5 convolutional blocks with progressive filter doubling
- Each block: Conv2D → BatchNormalization → ReLU → Conv2D → BatchNormalization → ReLU → MaxPooling2D

| Block | Filters | Output Size | Purpose |
|-------|---------|-------------|---------|
| 1 | 32 | 128×128 → 63×63 | Edge detection |
| 2 | 64 | 63×63 → 30×30 | Simple textures |
| 3 | 128 | 30×30 → 14×14 | Mid-level patterns |
| 4 | 256 | 14×14 → 6×6 | Complex shapes |
| 5 | 512 | 6×6 → 2×2 | High-level features |

**Classification Head**
- Flatten layer: 2×2×512 → 2048-d vector
- Dense layer: 1500 units → BatchNorm → ReLU → Dropout
- Output layer: 38 units (one per class) with Softmax activation

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 10
- **Batch Size**: 32
- **Image Preprocessing**: Resize to 128×128, normalization
- **Data Pipeline**: TensorFlow's `image_dataset_from_directory`

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.7+
pip package manager
```

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd smart-crop-disease-detection
```

2. **Install required packages**
```bash
pip install streamlit tensorflow numpy opencv-python matplotlib seaborn pandas scikit-learn kagglehub glob2
```

3. **Download the trained model**
- Ensure `trained_plant_disease_model.keras` is in the project directory
- Add a sample image `home_page.jpeg` for the homepage

4. **Run the application**
```bash
streamlit run main.py
```

5. **Access the web application**
- Open your browser and navigate to `http://localhost:8501`

## 💻 Usage

### Web Application Interface

The application provides three main pages:

#### 1. Home Page
- Welcome message and system overview
- Instructions on how to use the system
- Key features and benefits

#### 2. About Page
- Dataset information and statistics
- Technical details about the project
- Content breakdown (train/test/validation splits)

#### 3. Disease Recognition Page
- **Upload Image**: Choose a plant leaf image (JPG/PNG)
- **Preview**: View the uploaded image
- **Predict**: Get instant disease classification results
- **Validation**: Automatic accuracy checking for test images

### Using the Disease Recognition Feature

1. Navigate to the "Disease Recognition" page
2. Click "Choose an Image" and upload a plant leaf photo
3. Click "Show Image" to preview your upload
4. Click "Predict" to get the disease classification
5. View results with confidence indicators and validation status

## 📄 License

This project is developed for educational purposes as part of an AI course. Please ensure proper attribution when using or modifying the code.


**Note**: This system is designed for educational and research purposes. For critical agricultural decisions, please consult with agricultural experts and conduct additional validation.