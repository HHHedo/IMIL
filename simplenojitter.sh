CUDA_VISIBLE_DEVICES=1 python main.py --task DigestSegNojitter --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_27_Nojitter/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0  --workers 8 --database

CUDA_VISIBLE_DEVICES=1 python main.py --task DigestSegNojitter --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_27_Nojitter/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1  --workers 8 --database

CUDA_VISIBLE_DEVICES=1 python main.py --task DigestSegNojitter --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_27_Nojitter/f2/ --data_root /remote-home/share/DATA/HISMIL/5_folder/2  --workers 8 --database

CUDA_VISIBLE_DEVICES=1 python main.py --task DigestSegNojitter --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_27_Nojitter/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3  --workers 8 --database

CUDA_VISIBLE_DEVICES=1 python main.py --task DigestSegNojitter --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_27_Nojitter/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4  --workers 8 --database