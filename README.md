## 1. install mmdet
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

## 2. data convert

`python tools/data_convert.py`

```
├── data
│   └── Liu
│       ├── images
│       │   ├── annotations
│       │   ├── test
│       │   ├── train
│       │   └── val
│       └── labels
│           ├── train
│           └── val

```

## 3. train 

```
python tools/train.py cfg_file

For example:

➜  AL2LBH git:(master) ✗ python tools/train.py --config mmprojects/liu_detection/config/yolo.py     

```

## 4. results will be saved at `work_dirs`

## 5. inference single scene

```
For example
➜  AL2LBH git:(master) ✗ python demo/image_demo.py data/Liu/images/train/wired_001.jpg mmprojects/liu_detection/config/centernet.py --weight work_dirs/centernet/last_checkpoint --device cpu

inference results will be saved in `outputs`
```

[text](outputs/preds/wired_042.json)

more details in : https://mmdetection.readthedocs.io/zh-cn/latest/user_guides/index.html