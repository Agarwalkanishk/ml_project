# 🚀 Real-time Age and Gender Detection

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/Agarwalkanishk/ml_project?style=for-the-badge)](https://github.com/Agarwalkanishk/ml_project/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Agarwalkanishk/ml_project?style=for-the-badge)](https://github.com/Agarwalkanishk/ml_project/network)
[![GitHub issues](https://img.shields.io/github/issues/Agarwalkanishk/ml_project?style=for-the-badge)](https://github.com/Agarwalkanishk/ml_project/issues)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE) <!-- TODO: Add actual license file or specify MIT if it's default for open source -->

**A Python-based application for real-time age and gender estimation using deep learning models.**

</div>

## 📖 Overview

This project provides a straightforward Python script for performing real-time age and gender detection from images or video streams. It leverages OpenCV's Deep Neural Network (DNN) module with pre-trained Caffe models for age and gender classification, combined with an OpenCV-provided face detector. The application is designed for easy setup and execution, making it ideal for demonstrations, educational purposes, or as a foundational component for more complex computer vision systems.

## ✨ Features

-   🎯 **Face Detection:** Utilizes OpenCV's pre-trained DNN model for robust face detection in images or video frames.
-   👴👵 **Age Estimation:** Predicts the age group of detected faces using a Caffe deep learning model. Age categories include: `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`.
-   ♂️♀️ **Gender Estimation:** Classifies the gender (Male/Female) of detected faces using a separate Caffe deep learning model.
-   🖥️ **Real-time Processing:** Designed to process frames from a webcam or video file, displaying results overlaid on the video stream.
-   🖼️ **Image Processing:** Capable of processing static images to detect and classify faces.

## 🖥️ Screenshots

<!-- TODO: Add actual screenshots of the application running, showing detected faces with age and gender labels. -->
*Expected output showing bounding boxes, age, and gender labels on detected faces.*

## 🛠️ Tech Stack

**Runtime:**
[![Python](https://img.shields.io/badge/Python-3.x-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**Libraries:**
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.22.4-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

**Machine Learning:**
-   **Models:** Caffe (`.caffemodel`, `.prototxt`), OpenCV DNN (`.pb`, `.pbtxt`)
-   **Frameworks:** Deep Learning via OpenCV DNN module

## 🚀 Quick Start

Follow these steps to get the project up and running on your local machine.

### Prerequisites

-   **Python 3.x**
-   **pip** (Python package installer)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Agarwalkanishk/ml_project.git
    cd ml_project
    ```

2.  **Install dependencies**
    All required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

The `gad.py` script can be run with an image file as input or by default, it will attempt to use your webcam for real-time detection.

#### 1. Using a Static Image

To run the script on a specific image, provide the image file path as a command-line argument:

```bash
python gad.py --image <path_to_image_file>
# Example using provided sample images:
python gad.py --image girl1.jpg
python gad.py --image man1.jpg
python gad.py --image woman1.jpg
```

#### 2. Using Webcam (Real-time)

If no `--image` argument is provided, the script will automatically attempt to open your default webcam and perform real-time age and gender detection.

```bash
python gad.py
```
Press `q` to quit the video stream window.

## 📁 Project Structure

```
ml_project/
├── .gitignore                    # Standard Git ignore file
├── age_deploy.prototxt           # Caffe model architecture for age estimation
├── age_net.caffemodel            # Pre-trained Caffe model weights for age estimation
├── gad.py                        # Main Python script for age and gender detection
├── gender_deploy.prototxt        # Caffe model architecture for gender estimation
├── gender_net.caffemodel         # Pre-trained Caffe model weights for gender estimation
├── girl1.jpg                     # Sample image for demonstration
├── man1.jpg                      # Sample image for demonstration
├── minion.jpg                    # Sample image (non-human face for robustness testing)
├── opencv_face_detector.pbtxt    # OpenCV DNN model architecture for face detection
├── opencv_face_detector_uint8.pb # Pre-trained OpenCV DNN model weights for face detection
├── requirements.txt              # Python package dependencies
├── woman1.jpg                    # Sample image for demonstration
└── woman3.jpg                    # Sample image for demonstration
```

## ⚙️ Configuration

The application's core logic and model paths are primarily configured within `gad.py`.

### Models
-   **Face Detector Protobuf:** `opencv_face_detector.pbtxt`
-   **Face Detector Weights:** `opencv_face_detector_uint8.pb`
-   **Age Model Protobuf:** `age_deploy.prototxt`
-   **Age Model Weights:** `age_net.caffemodel`
-   **Gender Model Protobuf:** `gender_deploy.prototxt`
-   **Gender Model Weights:** `gender_net.caffemodel`

These paths are currently relative to the script's location.

### Parameters
-   **Confidence Threshold:** The minimum probability to filter weak face detections. This is typically hardcoded within the script to a value like `0.5` or `0.7`.
-   **Mean Subtraction Values:** Used for pre-processing images before feeding them into the neural networks, commonly `(104.0, 177.0, 123.0)` for models trained on ImageNet statistics.
-   **Scale Factor:** Used when resizing images for the neural networks, often `1.0`.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. <!-- TODO: Ensure a LICENSE file exists or update this section. -->

## 🙏 Acknowledgments

-   **OpenCV:** For providing robust computer vision tools and pre-trained models.
-   **Caffe Models:** The age and gender estimation models are commonly used pre-trained models in the computer vision community.

## 📞 Support & Contact

-   🐛 Issues: [GitHub Issues](https://github.com/Agarwalkanishk/ml_project/issues)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by [Agarwalkanishk](https://github.com/Agarwalkanishk)

</div>