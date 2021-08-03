# CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_19_NoFlip/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0 --workers 8  --ignore_thres 0.95 --mmt 0.5 --database 

# CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip  --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_19_NoFlip/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1 --workers 8  --ignore_thres 0.95 --mmt 0.5 --database 

# CUDA_VISIBLE_DEVICES=6 python main.py  --task DigestSegNoFlip --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_19_NoFlip/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2 --workers 8  --ignore_thres 0.95 --mmt 0.5 --database 

# CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip  --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_19_NoFlip/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3 --workers 8  --ignore_thres 0.95 --mmt 0.5 --database 

# CUDA_VISIBLE_DEVICES=6 python main.py  --task DigestSegNoFlip --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_19_NoFlip/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4 --workers 8  --ignore_thres 0.95 --mmt 0.5 --database 


CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20_NoFlip/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0  --workers 8 --database
CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20_NoFlip/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1  --workers 8 --database
CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20_NoFlip/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2  --workers 8 --database
CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20_NoFlip/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3  --workers 8 --database
CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20_NoFlip/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4  --workers 8 --database


CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_20_NoFlip/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0  --workers 8 --database

CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_20_NoFlip/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1  --workers 8 --database

CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_20_NoFlip/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2  --workers 8 --database

CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_20_NoFlip/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3  --workers 8 --database

CUDA_VISIBLE_DEVICES=6 python main.py --task DigestSegNoFlip --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_20_NoFlip/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4  --workers 8 --database