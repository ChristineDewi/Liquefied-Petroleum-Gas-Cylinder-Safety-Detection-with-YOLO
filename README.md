# Liquefied-Petroleum-Gas-Cylinder-Safety-Detection-with-YOLO

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
