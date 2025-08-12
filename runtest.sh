#python3 test.py --data data/data.yaml --cfg runs/train/yolov8n_float32/cfg.yaml --weights runs/train/yolov8n/train/yolo8n_train/weights/best.pt --name yolov8n_float32 --img-size 416 --batch-size 32 --device 0 --task test

python3 test.py --data data/data.yaml --cfg runs/train/quantyolov8_8w8a/cfg.yaml --weights runs/train/quantyolov8_8w8a/weights/best.pt --name quantyolov8_8w8a --img-size 416 --batch-size 32 --device 0 --task test
