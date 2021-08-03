# CUDA_VISIBLE_DEVICES=0 python main.py --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0  --workers 8 --database

# CUDA_VISIBLE_DEVICES=0 python main.py --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1  --workers 8 --database

# CUDA_VISIBLE_DEVICES=0 python main.py --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2 --workers 8 --database

# CUDA_VISIBLE_DEVICES=3 python main.py --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3 --workers 8 --database

# CUDA_VISIBLE_DEVICES=3 python main.py --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_20/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4 --workers 8 --database


CUDA_VISIBLE_DEVICES=6 python main.py  --config DigestSemi --log_dir ./experiments/NorAug_Semi/2021_07_29/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0  --workers 8 --database --semi_ratio 0.5


CUDA_VISIBLE_DEVICES=6 python main.py  --config DigestSemi --log_dir ./experiments/NorAug_Semi/2021_07_29/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1  --workers 8 --database --semi_ratio 0.5


CUDA_VISIBLE_DEVICES=6 python main.py  --config DigestSemi --log_dir ./experiments/NorAug_Semi/2021_07_29/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2  --workers 8 --database --semi_ratio 0.5


CUDA_VISIBLE_DEVICES=6 python main.py  --config DigestSemi --log_dir ./experiments/NorAug_Semi/2021_07_29/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3  --workers 8 --database --semi_ratio 0.5


CUDA_VISIBLE_DEVICES=6 python main.py  --config DigestSemi --log_dir ./experiments/NorAug_Semi/2021_07_29/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4  --workers 8 --database --semi_ratio 0.5