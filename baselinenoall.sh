CUDA_VISIBLE_DEVICES=4 python main.py --task DigestSegNoall  --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_28_Noall/f3/ --data_root /remote-home/share/DATA/HISMIL/5_folder/3 --workers 8 --database 

CUDA_VISIBLE_DEVICES=0 python main.py --task DigestSegNoall  --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_28_Noall/f0/ --data_root /remote-home/share/DATA/HISMIL/5_folder/0 --workers 8 --database 

CUDA_VISIBLE_DEVICES=1 python main.py --task DigestSegNoall  --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_28_Noall/f1/ --data_root /remote-home/share/DATA/HISMIL/5_folder/1 --workers 8 --database 

CUDA_VISIBLE_DEVICES=2 python main.py --task DigestSegNoall  --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_28_Noall/f2 --data_root /remote-home/share/DATA/HISMIL/5_folder/2 --workers 8 --database 

CUDA_VISIBLE_DEVICES=3 python main.py --task DigestSegNoall  --config DigestSeg --log_dir ./experiments/MoCoAug_Baseline/2020_07_28_Noall/f4/ --data_root /remote-home/share/DATA/HISMIL/5_folder/4 --workers 8 --database 
