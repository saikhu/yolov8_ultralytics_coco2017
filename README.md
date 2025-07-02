## YOLOv8 Training with COCO2017 - Object Detection (80 classes) - Ultralytics

### Venv prepare using Conda 
```bash
conda create -n yolov8 python=3.10 -y
conda activate yolov8
```

### Install the dependencies
Assuming you already that the Nvida drivers and toolkit installed.
```bash
# In your yolov8 env
pip install torch torchvision torchaudio
pip install ultralytics
```

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Data preprocessing
COCO2017 dataset comes with JSON format labels but YOLO accepts in indiviual label files in text format.
To convert the dataset you can you the following:
```python
from ultralytics.data.converter import convert_coco
# For COCO2017 Data conversion to YOLO Format using ultralytics
convert_coco(
     labels_dir="/media/usman/conti/DATASETS/COCO2017/annotations/instance",  # Directory containing your json file
     save_dir="/media/usman/conti/DATASETS/COCO2017/yolo_annotations",
     use_segments=False,
     use_keypoints=False,
     cls91to80=True,
     lvis=False
    )
```


### Training 

#### Quick 
```bash
from ultralytics.data.converter import convert_coco

# For keypoints data (like person_keypoints_val2017.json)
convert_coco(
    labels_dir="/media/usman/conti/DATASETS/COCO2017/annotations/",  # Directory containing your json file
    save_dir="/media/usman/conti/DATASETS/COCO2017/yolo_annotations/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
    lvis=False
)
```

#### Use `train.py`
Within Venv or docker container
`python train.py`

### Output 
The output of the model training is given in the `bash_log.txt` file.