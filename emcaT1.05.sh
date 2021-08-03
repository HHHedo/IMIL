
CUDA_VISIBLE_DEVICES=6 python main.py --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_30_T1.05/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0 --workers 8  --ignore_thres 1.05 --mmt 0.5 --database 

CUDA_VISIBLE_DEVICES=6 python main.py --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_30_T1.05/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1 --workers 8  --ignore_thres 1.05 --mmt 0.5 --database 

CUDA_VISIBLE_DEVICES=6 python main.py --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_30_T1.05/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2 --workers 8  --ignore_thres 1.05 --mmt 0.5 --database 

CUDA_VISIBLE_DEVICES=6 python main.py --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_30_T1.05/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3 --workers 8  --ignore_thres 1.05 --mmt 0.5 --database 

CUDA_VISIBLE_DEVICES=6 python main.py --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_30_T1.05/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4 --workers 8  --ignore_thres 1.05 --mmt 0.5 --database 