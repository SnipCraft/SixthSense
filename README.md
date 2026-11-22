# SixthSense

**An AI-powered assistive navigation prototype for blind and low-vision users.**

SixthSense is a research and development project exploring affordable, real-time obstacle awareness using computer vision, depth estimation, and directional haptic feedback.

The planned system combines a smartphone camera with an ESP32-based vibration belt. The camera captures the environment, AI models analyze objects and their relative depth, and directional vibration motors communicate obstacle information to the user.

> **Project status:** Active development. The current repository contains working computer-vision experiments, YOLO object detection, and Depth Anything V2 ONNX inference. The complete assistive navigation and hardware integration pipeline is still under development.

---

## 1. Project Goals

The long-term goals of SixthSense are:

- Detect objects in the user's surroundings.
- Estimate relative depth to understand which objects appear closer.
- Analyze obstacles in the left, center, and right regions.
- Prioritize nearby obstacles.
- Communicate directional warnings through vibration motors.
- Provide useful information through OCR and an AI-based question-answering component.
- Perform core processing locally, reducing dependence on cloud services.

## 2. Planned System Architecture

```text
             Smartphone Camera
                     |
                     v
              Camera Capture
                     |
          +----------+----------+
          |                     |
          v                     v
    YOLO11n Detection    Depth Anything V2
          |                     |
          +----------+----------+
                     |
                     v
          Spatial Obstacle Analysis
             Left / Center / Right
                     |
                     v
          Obstacle Prioritization
                     |
                     v
          Smartphone-to-ESP32 Link
                     |
                     v
             ESP32-WROOM
                     |
          +----------+----------+
          |          |          |
          v          v          v
       Left       Center      Right
      Vibration  Vibration   Vibration
        Motor      Motor       Motor
```

This is the intended architecture, not a claim that every component is already integrated.

## 3. Current Implementation

### Implemented and tested

- YOLO11 Nano webcam object-detection experiment.
- Depth Anything V2 Small (ViT-S) depth-estimation model.
- PyTorch-based depth-estimation experiments.
- ONNX export and inference.
- Fixed 392 x 392 ONNX model.
- Real-image comparison between PyTorch and ONNX outputs.
- Threaded webcam processing to keep camera preview responsive.
- CPU-based inference using ONNX Runtime.

### In development / planned

- Left, center, and right obstacle analysis.
- Obstacle prioritization and haptic decision logic.
- ESP32-WROOM communication and firmware.
- Three-direction vibration feedback.
- ML Kit OCR.
- Qwen2.5-0.5B question-answering component.
- Qualcomm ExecuTorch deployment and optimization.
- Complete smartphone application.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.11 |
| Object detection | YOLO11 Nano |
| Depth estimation | Depth Anything V2 Small |
| Deep learning | PyTorch |
| Optimized inference | ONNX Runtime |
| Model format | ONNX |
| Computer vision | OpenCV |
| Numerical processing | NumPy |
| Visualization | Matplotlib |
| Hardware target | ESP32-WROOM |
| Development platform | Windows |

The current development environment uses CPU inference. An NVIDIA GPU is not required for the current webcam experiments.

## 5. Repository Structure

```text
SixthSense/
|
|-- development/
|   |-- camera/
|   |   |-- webcam_test.py
|   |
|   |-- detection/
|   |   |-- yolo_webcam.py
|   |
|   |-- depth/
|       |-- Depth-Anything-V2/       # Git submodule
|       |-- benchmark_depth_onnx.py
|       |-- compare_real_image_392.py
|       |-- depth_anything_v2_vits.onnx
|       |-- depth_anything_v2_vits_392.onnx
|       |-- depth_anything_v2_vits_dynamic.onnx
|       |-- depth_test.py
|       |-- export_depth_onnx.py
|       |-- export_depth_onnx_392.py
|       |-- export_depth_onnx_dynamic.py
|       |-- live_depth_onnx_392.py
|       |-- live_depth_onnx_threaded.py
|       |-- live_depth_webcam.py
|       |-- validate_depth_onnx.py
|       |-- validate_dynamic_onnx.py
|       |-- validate_onnx_392.py
|       |-- test.jpg
|       |-- depth_output.png
|
|-- models/
|   |-- depth_anything_v2_vits.pth
|
|-- yolo11n.pt
|-- requirements.txt
|-- requirements-lock.txt
|-- .gitignore
|-- .gitattributes
|-- .gitmodules
|-- README.md
|
|-- esp32_firmware/       # Planned
|-- tests/                # Planned
|-- web_app/              # Planned
```

The repository may gain additional files as development continues.

## 6. Setup on a New Windows Computer

### Prerequisites

Install:

- Git
- Git LFS
- Python 3.11
- A webcam

Git LFS is required to download the large model files.

### Step 1: Clone the repository

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the URL of your GitHub repository.

```powershell
git lfs install

git clone --recurse-submodules <YOUR_GITHUB_REPOSITORY_URL>

cd SixthSense
```

If the repository was cloned without its submodule:

```powershell
git submodule update --init --recursive
```

### Step 2: Download Git LFS models

```powershell
git lfs pull
```

This downloads the large model files tracked by Git LFS.

### Step 3: Create a virtual environment

```powershell
py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

If PowerShell blocks environment activation, you can use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1
```

### Step 4: Install CPU-only PyTorch

Install the verified CPU versions:

```powershell
python -m pip install torch==2.15.0+cpu torchvision==0.29.2+cpu --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Install the remaining dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 6: Verify the environment

```powershell
python -c "import torch, torchvision, cv2, numpy, onnxruntime, ultralytics; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available()); print('OpenCV:', cv2.__version__); print('ONNX Runtime:', onnxruntime.__version__)"
```

Check installed dependencies:

```powershell
python -m pip check
```

## 7. Restore the Depth Anything V2 Checkpoint

The official Depth Anything V2 repository is included as a Git submodule.

The original checkpoint is not tracked inside that submodule. A backup copy is stored in the main repository through Git LFS:

```text
models/depth_anything_v2_vits.pth
```

Some existing scripts expect the checkpoint at:

```text
development/depth/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth
```

After cloning and downloading LFS files, restore it with:

```powershell
New-Item -ItemType Directory -Force .\development\depth\Depth-Anything-V2\checkpoints

Copy-Item .\models\depth_anything_v2_vits.pth .\development\depth\Depth-Anything-V2\checkpoints\depth_anything_v2_vits.pth
```

Verify that the checkpoint exists:

```powershell
Get-Item .\development\depth\Depth-Anything-V2\checkpoints\depth_anything_v2_vits.pth
```

The original model source is available at:

https://github.com/DepthAnything/Depth-Anything-V2

## 8. Running the Current Experiments

Run commands from the main `SixthSense` project directory.

### A. Test the webcam

```powershell
python .\development\camera\webcam_test.py
```

### B. Run YOLO11 Nano object detection

```powershell
python .\development\detection\yolo_webcam.py
```

This loads `yolo11n.pt` from the project root and performs webcam object detection.

### C. Run Depth Anything V2 with ONNX Runtime

```powershell
python .\development\depth\live_depth_onnx_392.py
```

This uses the fixed 392 x 392 ONNX model.

### D. Run threaded ONNX depth estimation

```powershell
python .\development\depth\live_depth_onnx_threaded.py
```

This runs inference in a background thread so that camera capture and display can continue while depth estimation is processing.

Press **Q** to exit the webcam windows.

## 9. Available Depth Models

| Model | Purpose |
|---|---|
| `depth_anything_v2_vits.onnx` | Original 518 x 518 ONNX export |
| `depth_anything_v2_vits_392.onnx` | Fixed 392 x 392 ONNX export |
| `depth_anything_v2_vits_dynamic.onnx` | Experimental dynamic-shape export |
| `depth_anything_v2_vits.pth` | Original PyTorch checkpoint |

**Important:** The experimental dynamic ONNX export has shown shape-related errors at resolutions other than 518 x 518. Use the fixed 392 x 392 model for the current threaded webcam implementation.

## 10. Model and Performance Notes

The current development work has compared PyTorch and ONNX inference on the same CPU-based computer.

For the fixed 392 x 392 model, the observed benchmark results were approximately:

| Metric | PyTorch | ONNX Runtime |
|---|---:|---:|
| Median inference time | 0.544 s | 0.381 s |
| Median inference FPS | 1.84 | 2.63 |

These are measurements from one development machine, not guaranteed performance on other devices.

### Important depth limitation

Depth Anything V2 estimates **relative depth**, not calibrated physical distance in metres.

Its output can help estimate which regions appear closer or farther away, but it should not be treated as a direct measurement of obstacle distance without additional calibration or sensing.

## 11. Troubleshooting

### Webcam cannot open

Check whether another application is using the webcam. You can also check whether the camera index needs to be changed.

### ONNX model not found

Verify that the ONNX model exists in:

```text
development/depth/
```

### Depth Anything checkpoint not found

Restore the checkpoint using the instructions in Section 7.

### Depth Anything module not found

Initialize the submodule:

```powershell
git submodule update --init --recursive
```

### Model files appear as small text files

The Git LFS files may not have been downloaded. Run:

```powershell
git lfs install

git lfs pull
```

### Dependency errors

Activate the virtual environment and run:

```powershell
python -m pip check
```

## 12. Safety Disclaimer

SixthSense is an experimental assistive-technology prototype. It has not been validated as a safety-critical mobility device.

It should not replace a white cane, guide dog, orientation and mobility training, or other established mobility aids.

## 13. Project Vision

The long-term vision is to develop an affordable, locally processed, multimodal assistive navigation system that combines computer vision, relative depth estimation, intelligent obstacle prioritization, and directional haptic feedback.

---

**SixthSense — Helping users perceive their surroundings through intelligent assistance.**
