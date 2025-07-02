from ultralytics import YOLO

# Path to the YOLO model (pretrained weights)
model = YOLO('yolov8n.pt')  # You can use yolov8s.pt, yolov8m.pt, etc.

# Path to your dataset YAML (should point to your coco2017.yaml)
DATASET_PATH = '/srv/DATA/DATASETS/COCO2017/YOLO_format'
YAML_PATH = '/srv/DATA/DATASETS/COCO2017/YOLO_format/coco2017.yaml'  # Adjust if you named/located it differently

# Start training
results = model.train(
    data=YAML_PATH,     # path to data yaml
    epochs=2,
    imgsz=640,
    batch=128,           # adjust to fit your GPU memory
    device=[0, 1, 2, 3], # which GPU (single GPU device=N N=the index of the gpu)
    workers=20,          # adjust based on CPU cores
    project='runs',     # output folder
    name='yolov8_coco_multiple_gpus'  # experiment name
)
# Results object has all training metrics
# print(results)
