```py
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```
用以上代码即可简单的执行。

其中yolo11n.pt是预训练的模型，节省时间。

其中主要知识在：
<a href="https://www.runoob.com/html/html-links.html#tips">yolov11使用</a>