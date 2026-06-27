# Liquefied-Petroleum-Gas-Cylinder-Safety-Detection-with-YOLO

Install the original yolov12 github:
https://github.com/sunsmarterjie/yolov12 

<img width="1094" height="432" alt="image" src="https://github.com/user-attachments/assets/d50873a1-6b9c-4991-a6f7-c35ae3679b2b" />

conda create -n yolov12 python=3.11 supervision flash-attn
conda activate yolov12
git clone https://github.com/sunsmarterjie/yolov12 && cd yolov12
pip install -r requirements.txt
pip install -e .
