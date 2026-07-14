# utility params
batch_size = 2
theta = 0.674  #
gamma = 0.72 #
lr = 0.00001  #
lr_ae = 0.00001
weight_decay=0.0000009
# requirments

# Hyper-parameters for this UDA Implementation 1
alpha = 0.80   # GRL Constant   0.1
beta = 0.42   # Weighting factor for Domain loss    0.12/SGD
theta = 0.482  # Weighting factor for Contrastive loss   0.2 
# Optimizaer and its parameter
lr_x = 0.01
weight_decay_x = 0.00001
# feature_dim = 128 / Z-Space vector dim
# temperature = 0.4   # Contrastive loss
# debiased = True     # Contrastive loss
# tau_plus=0.05        # Contrastive loss
# optimizer = SGD
# image size and features size
# input_image_size = [4, 3, 608, 608], [4, 3, 1280, 1280] / dataloader.py / target path + discriminator_grl.py + 
# base_model_output = [4, 512, 19, 19], [4, 512, 40, 40] / contrastive.py
# P3_Model_output = [4, 128, 76, 76], [4, 128, 160, 160] + cfg_path
# P4_Model_output = [4, 256, 38, 38], [4, 256, 80, 80] + cfg_path
# P5_Model_output = [4, 512, 19, 19], [4, 512, 40, 40] + cfg_path
# Correcting the dataloader 
# Target folder with a target image must be created to pass
# Target_loader file path 
# Modifying the .yaml file


"""
BASELINE BENCHMARKING
https://github.com/ultralytics/yolov5/issues/318

"""

'''
https://github.com/ultralytics/yolov5/issues/8191
https://stackoverflow.com/questions/73439716/how-to-use-val-py-to-calculate-map-on-custom-testing-dataset-of-yolov5
https://github.com/WongKinYiu/yolov7/issues/395
https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/compare_YOLOv7_vs_YOLOv5s6.ipynb
https://github.com/ultralytics/yolov5/issues/11984

python val.py --task test --single-cls --data s2c.yaml --img 608 --batch 1 --conf 0.001 --iou 0.5 --device 1 --weights runs/train/KIT_BASE/weights/best.pt --name KIT_BASE
...
Speed: 0.7ms pre-process, 18.7ms inference, 1.7ms NMS per image at shape (1, 3, 1280, 1280)
...
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.449

https://towardsai.net/p/machine-learning/yolov5m-implementation-from-scratch-with-pytorch

python train_123.py --img 608 --cfg yolov5s.yaml --device 1 --hyp hyp.scratch-low.yaml --batch 2 --epochs 50 --optimizer SGD --data sim10k.yaml --weights C:/Users/anduw/Desktop/my_project/C-PAT/yolov5/runs/train/SIM_INT/weights/best.pt --workers 0 --name S2C_001
python train.py --img 1280 --cfg yolov5s.yaml --freeze 10 --device 1 --hyp hyp.no-augmentation.yaml --batch 4 --epochs 20 --optimizer SGD --data kitti.yaml --weights C:/Users/anduw/Desktop/my_project/ASST/yolov5/runs/train/K2C_001/weights/best.pt --workers 0 --name K2C_RET_001

python val.py --task speed --img 640 --single-cls --device 1 --batch 1 --iou-thres 0.45 --conf-thres 0.25 --data C:/Users/anduw/Desktop/my_project/C-PAT/yolov5/data/s2c.yaml --weights C:/Users/anduw/Desktop/my_project/C-PAT/yolov5/runs/train/SIM_INT/weights/best.pt --workers 0 --name BS_CITY2FOGGY_001
python detect.py --source /home/andualemwelabo.tulu/C-PAT/yolov5/data/img --weights /home/andualemwelabo.tulu/C-PAT/yolov5/runs/train/KITTI_INT/weights/best.pt --conf 0.5 --img-size 608

python val.py --task test --single-cls --device 1 --iou-thres 0.65 --weights  --data s2c.yaml --img 608 --batch 4 --conf 0.001 --workers 0 --name BS_C

python benchmarks.py --device 0 --data kitti2city.yaml --weights /home/andualemwelabo.tulu/C-PAT/yolov5/runs/train/K2C_009/weights/best.pt --img 640 --test

https://github.com/ultralytics/yolov5/issues/1314 (--freeze 10 or --freeze 24, ADDA)

python val.py --task test --single-cls --data s2c.yaml --img 608 --batch 1 --conf 0.25 --iou 0.5 --device 1 --weights C:/Users/anduw/Desktop/my_project/ASST/yolov5/runs/train/SIM_INT/weights/best.pt --name BASE_LINE

python val.py --weights C:/Users/anduw/Desktop/my_project/ASST/yolov5/runs/train/K2C_001/weights/best.pt --data s2f.yaml --img 608 


import wandb 
os.environ["WANDB_DISABLED"]='true' 
wandb.disabled=True


python train.py --img 640 --cfg yolov5s.yaml --device 1 --hyp hyp.scratch-low.yaml --batch 4 --epochs 30 --optimizer SGD --data kitti.yaml --weights /home/andualemwelabo.tulu/expt_01/yolov5/yolov5s.pt --workers 0 --name KITTI_INT

python raw_train.py --img 640 --cfg yolov5s.yaml --device 1 --hyp hyp.scratch-low.yaml --batch 1 --epochs 30 --optimizer SGD --data sim.yaml --weights /home/andualemwelabo.tulu/expt_01/yolov5/runs/train/SIM_INT/weights/best.pt --workers 0 --name ZAK_001

python val.py --task test --weights /home/andualemwelabo.tulu/expt_01/yolov5/runs/train/CITY_001/weights/best.pt --data /home/andualemwelabo.tulu/expt_01/yolov5/data/c2f.yaml --img 640 --iou-thres 0.5 --conf 0.001


'''


