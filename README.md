# Liquefied-Petroleum-Gas-Cylinder-Safety-Detection-with-YOLO

Abstract

Background. Liquefied Petroleum Gas (LPG) cylinders are widely used in domestic and industrial environments, but improper handling, unsafe placement, damaged regulators, and leakage-related conditions may increase the risk of fire and explosion. Computer vision-based detection offers a practical approach for supporting early safety inspection. 
Methods. This study evaluates the performance of several YOLO-based deep learning models for LPG cylinder safety detection and investigates the effect of Contrast Limited Adaptive Histogram Equalization (CLAHE) as an image enhancement technique. Four lightweight YOLO models, namely YOLOv11n, YOLOv11s, YOLOv12n, and YOLOv12s, were trained and tested using both the original dataset and the CLAHE-enhanced dataset. Results. The experimental results show that CLAHE improves the overall average performance of the models. In the training phase, the average performance increased slightly from 94.10% on the original dataset to 94.15% using CLAHE. A more noticeable improvement was observed during testing, where the average performance increased from 95.55% to 95.77%. The best test result was achieved by YOLOv12s with CLAHE, reaching 96.10%. These findings indicate that CLAHE can enhance image contrast and improve feature visibility, which contributes positively to LPG cylinder safety detection. Although the improvement is not consistent across all individual models, the overall test performance demonstrates that preprocessing plays an important role in improving detection reliability. This study provides a comparative baseline for YOLO-based LPG cylinder safety detection and supports the future development of intelligent monitoring systems for gas cylinder safety applications.
Keywords: Liquefied Petroleum Gas; LPG cylinder detection; YOLOv11; YOLOv12; CLAHE preprocessing; deep learning; computer vision; safety detection

# Code Information
Install the original yolov12 github:
https://github.com/sunsmarterjie/yolov12 

<img width="1094" height="432" alt="image" src="https://github.com/user-attachments/assets/d50873a1-6b9c-4991-a6f7-c35ae3679b2b" />

# Installation

conda create -n yolov12 python=3.11 supervision flash-attn
conda activate yolov12
git clone https://github.com/sunsmarterjie/yolov12 && cd yolov12
pip install -r requirements.txt
pip install -e .

# Dataset

Download the dataset: https://zenodo.org/records/20756494

<img width="797" height="537" alt="image" src="https://github.com/user-attachments/assets/817bcf98-0154-4511-b00c-0a8466a01329" />

# Training and Testing

from ultralytics import YOLO

model = YOLO('yolov12n.yaml')

# Train the model
results = model.train(
  data='coco.yaml',
  epochs=600, 
  batch=256, 
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1,2,3",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()


<img width="981" height="342" alt="image" src="https://github.com/user-attachments/assets/2b075577-ccad-4ad0-9406-30a2e632a347" />



@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
