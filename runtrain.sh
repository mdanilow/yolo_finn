# python -m torch.distributed.launch --nproc_per_node 2 --master_port 9528 train.py --resume runs/train/yololit_320_6x_new2/weights/last.pt
# python train.py --data data/coco8.yaml --cfg runs/train/yololit_320_6x_new4/cfg.yaml --hyp data/hyp.scratch.p5_requant.yaml --name continue_test --batch-size 16 --epochs 1 --device 0
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 9532 train.py --cfg cfg/training/yololit_320_4w4a.yaml --weights runs/train/yololit_320_6x_new_continue_0epoch_3600/weights/best.pt --hyp data/hyp.scratch.p5_requant4.yaml --name yololit_320_4w4a_requant4_adam_ema --adam --fresh_optimizer --batch-size 32 --epochs 1800 --device 0,1 --sync-bn
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 9532 train.py --cfg cfg/training/yololit_320_4w4a.yaml --weights runs/train/yololit_320_4w4a_requant4_adam/weights/best.pt --hyp data/hyp.scratch.p5_requant4.yaml --name yololit_320_4w4a_requant4_adam_uadetrac_nc4 --data data/ua-detrac.yaml --freeze 17 --adam --fresh_optimizer --batch-size 32 --epochs 1800 --device 0,1 --sync-bn
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 9532 train.py --cfg cfg/training/quantyolov8_4w4a.yaml --weights runs/train/v8_float_3202/weights/best.pt --hyp data/hyp_v8_finetune.yaml --name quantyolov8_4w4a_independent_acts --img-size 320 320 --batch-size 32 --epochs 600 --device 0,1 --sync-bn
# python -m torch.distributed.launch --nproc_per_node 2 --master_port 9533 train.py --cfg cfg/training/quantyolov8_4w4a_common_act.yaml --weights runs/train/v8_float_3202/weights/best.pt --hyp data/hyp_v8_finetune.yaml --name quantyolov8_4w4a_a4_comact_fropt_frbn --img-size 320 320 --batch-size 32 --epochs 900 --device 0,1 --sync-bn --freeze_bn_stats --fresh_optimizer --fresh_ema --adam

# python train.py --cfg cfg/training/quantyolov8_8w8a.yaml --weights runs/train/v8_float_320/weights/best.pt --hyp data/hyp_v8_finetune_lre3.yaml --name quantyolov8_8w8a --img-size 320 320 --fresh_ema --batch-size 32 --epochs 900 --device 0
# python train.py --cfg cfg/training/quantyolov8_4w4a.yaml --weights runs/train/v8_float_320/weights/best.pt --hyp data/hyp_v8_finetune_lre3.yaml --name quantyolov8_4w4a --img-size 320 320 --fresh_ema --batch-size 32 --epochs 900 --device 0
#python train.py --cfg cfg/training/quantyolov8_4w4a.yaml --weights runs/train/v8_float_320/weights/best.pt --hyp data/hyp_v8_finetune_lre3.yaml --name quantyolov8_4w4a_freezebn --img-size 320 320 --fresh_ema --batch-size 32 --epochs 900 --device 0 --freeze_bn_stats


# Suraj Experiments
#python3 train.py --data data/data.yaml --cfg cfg/training/quantyolov8_8w8a.yaml --weights runs/train/yolov8n/train/yolo8n_train/weights/best.pt --hyp data/hyp_v8_finetune_lre3.yaml --name quantyolov8_8w8a_416 --img-size 416 416 --fresh_ema --batch-size 32 --epochs 900 --device 0

#python3 train.py --data data/data.yaml --cfg cfg/training/quantyolov8_8w8a.yaml --weights runs/train/yolov8n_hpd2_modified_augmented/train/yolo8n_train/weights/best.pt --hyp data/hyp_v8_finetune_lre3.yaml --name quantyolov8_8w8a_640 --img-size 640 640 --fresh_ema --batch-size 32 --epochs 900 --device 0

#python3 train.py --data data/data.yaml --cfg cfg/training/quantyolov8_8w8a.yaml --weights runs/train/yolov8n_hpd3_augmented_320/train/yolo8n_train/weights/best.pt --hyp data/hyp_v8_finetune_lre3.yaml --name quantyolov8_8w8a_320 --img-size 320 320 --fresh_ema --batch-size 32 --epochs 900 --device 0

#python3 train.py --data data/data.yaml --cfg cfg/training/quantyolov8_8w8a.yaml --weights runs/train/yolo8n_hpd2_modified_aug_640/weights/best.pt --hyp data/hyp_v8_finetune_lre3.yaml --name quantyolov8_8w8a_640_new --img-size 640 640 --fresh_ema --batch-size 32 --epochs 900 --device 0

