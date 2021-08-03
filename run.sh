# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_6_29/f1/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/1 --workers 8 --ignore_thres 0.95 --mmt 0

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_6_29/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2 --workers 8 --ignore_thres 0.95 --mmt 0

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_6_29/f3/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8 --ignore_thres 0.95 --mmt 0
# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_6_29/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4 --workers 8 --ignore_thres 0.95 --mmt 0

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_7_7_mmt0.5/f1/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/1 --workers 8 --ignore_thres 0.95 --mmt 0.5

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_7_7_mmt0.5/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2 --workers 8 --ignore_thres 0.95 --mmt 0.5

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_7_7_mmt0.5/f3_WeightedBCE/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8 --ignore_thres 0.95 --mmt 0.5

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_7_7_mmt0.5/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4 --workers 8 --ignore_thres 0.95 --mmt 0.5

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_7_7_mmt0.75/f0/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/0 --workers 8 --ignore_thres 0.95 --mmt 0.75

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_7_7_mmt0.75/f1_WeightedBCE/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/1 --workers 8 --ignore_thres 0.95 --mmt 0.75

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_7_7_mmt0.75/f2_WeightedBCE/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2 --workers 8 --ignore_thres 0.95 --mmt 0.75

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_7_7_mmt0.75/f3_WeightedBCE/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8 --ignore_thres 0.95 --mmt 0.75

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/EMCA_thres_0.95_stable/2021_7_7_mmt0.75/f4_WeightedBCE/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4 --workers 8 --ignore_thres 0.95 --mmt 0.75


python main.py --config DigestSeg --log_dir ./experiments/Baseline_NormalAug/2021_07_11/f0/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/0

python main.py --config DigestSeg --log_dir ./experiments/Baseline_NormalAug/2021_07_11/f1/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/1

python main.py --config DigestSeg --log_dir ./experiments/Baseline_NormalAug/2021_07_11/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2

python main.py --config DigestSeg --log_dir ./experiments/Baseline_NormalAug/2021_07_11/f3/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/3

python main.py --config DigestSeg --log_dir ./experiments/Baseline_NormalAug/2021_07_11/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4




# python main.py --config DigestSegFull --log_dir ./experiments/Full_NormalAug/2020_07_11/f0/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/0  --workers 8 

# python main.py --config DigestSegFull --log_dir ./experiments/Full_NormalAug/2020_07_11/f1/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/1  --workers 8 

# python main.py --config DigestSegFull --log_dir ./experiments/Full_NormalAug/2020_07_11/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2  --workers 8 

# python main.py --config DigestSegFull --log_dir ./experiments/Full_NormalAug/2020_07_11/f3/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8 

# python main.py --config DigestSegFull --log_dir ./experiments/Full_NormalAug/2020_07_11/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4  --workers 8 

# CUDA_VISIBLE_DEVICES=0 python main.py --config DigestSegRCE --log_dir ./experiments/NorAug_RCE/2020_07_14/f0/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/0 --workers 8 --mmt 0.75

# CUDA_VISIBLE_DEVICES=0 python main.py --config DigestSegRCE --log_dir ./experiments/NorAug_RCE/2020_07_14/f1/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/1 --workers 8 --mmt 0.75

# CUDA_VISIBLE_DEVICES=0 python main.py --config DigestSegRCE --log_dir ./experiments/NorAug_RCE/2020_07_14/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2 --workers 8 --mmt 0.75




# CUDA_VISIBLE_DEVICES=0 python main.py --config DigestSegRCE --log_dir ./experiments/NorAug_RCE/2020_07_14/f3/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8 --mmt 0.75

# CUDA_VISIBLE_DEVICES=0 python main.py --config DigestSegRCE --log_dir ./experiments/NorAug_RCE/2020_07_14/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4 --workers 8 --mmt 0.75


# python main.py --config DigestSegPB --log_dir ./experiments/NorAug_PB/2021_07_14/f0/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/0 --workers 8  --mmt 0
# python main.py --config DigestSegPB --log_dir ./experiments/NorAug_PB/2021_07_14/f1/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/1 --workers 8  --mmt 0
# python main.py --config DigestSegPB --log_dir ./experiments/NorAug_PB/2021_07_14/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2 --workers 8  --mmt 0
# python main.py --config DigestSegPB --log_dir ./experiments/NorAug_PB/2021_07_14/f3/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8  --mmt 0
# python main.py --config DigestSegPB --log_dir ./experiments/NorAug_PB/2021_07_14/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4 --workers 8  --mmt 0

# python main.py --task DigestSeg --config DigestSegTOPK --log_dir ./experiments/NorAug_TOPK/2021_07_14/f0/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/0  --workers 8 --mmt 0
# python main.py --task DigestSeg --config DigestSegTOPK --log_dir ./experiments/NorAug_TOPK/2021_07_14/f1/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/1  --workers 8 --mmt 0
# python main.py --task DigestSeg --config DigestSegTOPK --log_dir ./experiments/NorAug_TOPK/2021_07_14/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2  --workers 8 --mmt 0
# python main.py --task DigestSeg --config DigestSegTOPK --log_dir ./experiments/NorAug_TOPK/2021_07_14/f3/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/3  --workers 8 --mmt 0
# python main.py --task DigestSeg --config DigestSegTOPK --log_dir ./experiments/NorAug_TOPK/2021_07_14/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4  --workers 8 --mmt 0

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_19/f0/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/0 --workers 8  --ignore_thres 0.95 --mmt 0.5 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_19/f1/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/1 --workers 8  --ignore_thres 0.95 --mmt 0.5 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_19/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2 --workers 8  --ignore_thres 0.95 --mmt 0.5 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_19/f3/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8  --ignore_thres 0.95 --mmt 0.5 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_19/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4 --workers 8  --ignore_thres 0.95 --mmt 0.5 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_20_mmt0.75/f0/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/0 --workers 8  --ignore_thres 0.95 --mmt 0.75 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_20_mmt0.75/f1/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/1 --workers 8  --ignore_thres 0.95 --mmt 0.75 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_20_mmt0.75/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2 --workers 8  --ignore_thres 0.95 --mmt 0.75 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_20_mmt0.75/f3/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8  --ignore_thres 0.95 --mmt 0.75 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_20_mmt0.75/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4 --workers 8  --ignore_thres 0.95 --mmt 0.75 


# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_20_mmt0/f0/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/0 --workers 8  --ignore_thres 0.95 --mmt 0

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_20_mmt0/f1/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/1 --workers 8  --ignore_thres 0.95 --mmt 0

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_20_mmt0/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2 --workers 8  --ignore_thres 0.95 --mmt 0 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_20_mmt0/f3/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8  --ignore_thres 0.95 --mmt 0

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_20_mmt0/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4 --workers 8  --ignore_thres 0.95 --mmt 0 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_23_mmt0.5/f0/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/0 --workers 8  --ignore_thres 0.95 --mmt 0.5

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_23_mmt0.5/f1/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/1 --workers 8  --ignore_thres 0.95 --mmt 0.5

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_23_mmt0.5/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2 --workers 8  --ignore_thres 0.95 --mmt 0.5 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_23_mmt0.5/f3/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8  --ignore_thres 0.95 --mmt 0.5

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_23_mmt0.5/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4 --workers 8  --ignore_thres 0.95 --mmt 0.5



# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_25_mmt0.9/f0/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/0 --workers 8  --ignore_thres 0.95 --mmt 0.9

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_25_mmt0.9/f1/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/1 --workers 8  --ignore_thres 0.95 --mmt 0.9

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_25_mmt0.9/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2 --workers 8  --ignore_thres 0.95 --mmt 0.9 

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_25_mmt0.9/f3/ --data_root  /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8  --ignore_thres 0.95 --mmt 0.9

# python main.py --config DigestSegEMCAV2 --log_dir ./experiments/NorAug_EMCA/2020_07_25_mmt0.9/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4 --workers 8  --ignore_thres 0.95 --mmt 0.9

# python main.py --config DigestSemi --log_dir ./experiments/NorAug_semi/2020_07_26/f0/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/0 --workers 8  --semi_ratio 0.5

# python main.py --config DigestSemi --log_dir ./experiments/NorAug_semi/2020_07_26/f1/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/1 --workers 8  --semi_ratio 0.5
# python main.py --config DigestSemi --log_dir ./experiments/NorAug_semi/2020_07_26/f2/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/2 --workers 8  --semi_ratio 0.5
# python main.py --config DigestSemi --log_dir ./experiments/NorAug_semi/2020_07_26/f3/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/3 --workers 8  --semi_ratio 0.5
# python main.py --config DigestSemi --log_dir ./experiments/NorAug_semi/2020_07_26/f4/ --data_root /home/ltc/Documents/ltc/DATA/HISMIL/4 --workers 8  --semi_ratio 0.5
    