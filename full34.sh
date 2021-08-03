CUDA_VISIBLE_DEVICES=4 python main.py --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_30_res34/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0  --workers 8 --database --backbone res34

CUDA_VISIBLE_DEVICES=4 python main.py --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_30_res34/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1  --workers 8 --database  --backbone res34

CUDA_VISIBLE_DEVICES=4 python main.py --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_30_res34/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2 --workers 8 --database --backbone res34

CUDA_VISIBLE_DEVICES=4 python main.py --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_30_res34/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3 --workers 8 --database  --backbone res34

CUDA_VISIBLE_DEVICES=4 python main.py --config DigestSegFull --log_dir ./experiments/MoCoAug_Full/2021_07_30_res34/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4 --workers 8 --database  --backbone res34
