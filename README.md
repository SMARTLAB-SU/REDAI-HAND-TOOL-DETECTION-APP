# REDAI Hand Tool Detection & Evaluation App

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%2Fv11-green.svg)](https://github.com/ultralytics/ultralytics)

A real-time hand tool detection, segmentation, and automated evaluation framework powered by YOLO object detection models and Intel RealSense depth sensing.

---

## 📁 Project Structure

```text
project-root/
├── Dataset/                  # Dataset documentation, reports, and annotations
│   ├── README.md             # Dataset source, format, and preprocessing guide
│   └── Report/               # Dataset analysis reports and EDA artifacts
├── App/                      # Main application binaries, source code, and installer scripts
│   ├── App.exe               # Compiled application executable (placeholder)
│   ├── Source Code/          # Application Python source files
│   │   ├── main.py           # Main GUI application entry point
│   │   └── Additional Files/ # Configuration, assets, and helper resources
│   └── ISS/                  # Inno Setup Script (.iss) for creating Windows installers
├── Output/                   # Model inference results and visualization outputs
│   └── README.md             # Description of output folder contents

├── Evaluation/               # Model inference and benchmarking scripts
│   └── inference.py          # Command-line inference & evaluation engine
├── Documentation/            # Technical documentation and presentation slides
│   ├── Comprehensive_Doc/    # In-depth technical documentation
│   └── PPT/                  # Presentation slides and project decks
├── .gitignore                # Git ignore rules for Python, builds, and media outputs
├── README.md                 # Top-level project overview and instructions
└── requirements.txt          # Python package dependencies
```

---

## ⚡ Quick Start

### 0. Download Standalone Windows Application (`App.exe`)

- 🚀 **Download Application Executable:** [Download App.exe from Google Drive](https://drive.google.com/file/d/1K2EkOdG_1VfzNfkzcvB0EkTEIUGKXVhO/view?usp=drive_link)

### 1. Installation


Clone the repository and install the required dependencies:

```bash
git clone https://github.com/SMARTLAB-SU/REDAI-HAND-TOOL-DETECTION-APP.git
cd REDAI-HAND-TOOL-DETECTION-APP
pip install -r requirements.txt
```

### 2. Running Model Inference & Evaluation

To run model evaluation against sample images or datasets:

```bash
# Run inference on a single image and compare with Ground Truth
python Evaluation/inference.py --model "App/Source Code/u_best.pt" --source sample.jpg --company Kinchrome

# Run batch inference on a test directory
python Evaluation/inference.py --model "App/Source Code/u_best.pt" --source Dataset/Images/test --gt-excel "App/Source Code/Book1.xlsx"
```

---

## 🧰 Supported Classes (14 Hand Tools)

1. Adaptor
2. Allen Key
3. Bit Holder
4. Bit Sockets
5. Bits
6. Deep Socket
7. Extension Bar
8. Flex Handle
9. Rachet-Handels
10. S Handle
11. Sockets
12. Spanners
13. long bit socket
14. universal joint

---

## 🎬 Output & Demonstration

- **Inference Video Demo:** [Watch Demonstration Video on Google Drive](https://drive.google.com/file/d/1kNv3vJCNFQUKjNTvZONUK_i5J90awXEC/view?usp=drive_link)


---

## 📜 License

Distributed under the MIT License.

