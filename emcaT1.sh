
CUDA_VISIBLE_DEVICES=3 python main.py --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_30_T1/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0 --workers 8  --ignore_thres 1 --mmt 0.5 --database 

CUDA_VISIBLE_DEVICES=3 python main.py --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_30_T1/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1 --workers 8  --ignore_thres 1 --mmt 0.5 --database 

CUDA_VISIBLE_DEVICES=3 python main.py --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_30_T1/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2 --workers 8  --ignore_thres 1 --mmt 0.5 --database 

CUDA_VISIBLE_DEVICES=3 python main.py --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_30_T1/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3 --workers 8  --ignore_thres 1 --mmt 0.5 --database 

CUDA_VISIBLE_DEVICES=3 python main.py --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_30_T1/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4 --workers 8  --ignore_thres 1 --mmt 0.5 --database 