# CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_19_NoGray/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0 --workers 8  --ignore_thres 0.95 --mmt 0.5 --database 

# CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray  --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_19_NoGray/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1 --workers 8  --ignore_thres 0.95 --mmt 0.5 --database 

# CUDA_VISIBLE_DEVICES=2 python main.py  --task DigestSegNoGray --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_19_NoGray/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2 --workers 8  --ignore_thres 0.95 --mmt 0.5 --database 

# CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray  --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_19_NoGray/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3 --workers 8  --ignore_thres 0.95 --mmt 0.5 --database 

# CUDA_VISIBLE_DEVICES=2 python main.py  --task DigestSegNoGray --config DigestSegEMCAV2 --log_dir ./experiments/MoCoAug_EMCA/2020_07_19_NoGray/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4 --workers 8  --ignore_thres 0.95 --mmt 0.5 --database 

CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20_NoGray/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0  --workers 8 --database
CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20_NoGray/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1  --workers 8 --database
CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20_NoGray/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2  --workers 8 --database
CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20_NoGray/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3  --workers 8 --database
CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20_NoGray/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4  --workers 8 --database


CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_20_NoGray/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0  --workers 8 --database

CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_20_NoGray/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1  --workers 8 --database

CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_20_NoGray/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2  --workers 8 --database

CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_20_NoGray/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3  --workers 8 --database

CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoGray --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_20_NoGray/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4  --workers 8 --database