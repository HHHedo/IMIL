## Interventional Multi-Instance Learning with Deconfounded Instance-Level Prediction

Implementation of IMIL.

### Environments

- **OS**: Ubuntu18.04.4 LTS
- **GCC/G++**: 7.5.0
- **GPU**: GeForce GTX TITAN X *1 |12G Memory
- **CUDA Version**: 10.1
- **NVIDIA-SMI**: 450.57
- **CPU**: Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz * 12.

### Installation

Install requirements:
```
pip install -r requirements.txt
```

For [faiss](https://github.com/facebookresearch/faiss), we recommend installing with the following command:

```
pip install faiss-gpu
```

See [faiss-wheels](https://github.com/kyamagu/faiss-wheels) for more details.

For [PyTorch](https://pytorch.org/) , we recommend following the instruction on official website.

### Data Folder Structure

We support training and testing on [DigestPath](https://digestpath2019.grand-challenge.org/), [Camelyon16](https://camelyon16.grand-challenge.org/)and [Pascal VOC 07](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/). Their data roots are like:

```
Data Root/
	DigestPath/
		five_folder/
			f0/
				train/
					pos/
						WSI_0/
							WSI_O_X_Y_(instance_label,bag_label).png
				        neg/
				test/
            	
	Camenlyon16/
		train/
			pos_instances_group_in_pos_bags/
				WSI_0/
				WSI_3/
   		 	neg_instances_group_in_pos_bags/
				WSI_0/
				WSI_3/
            		neg_instances_group_in_neg_bags/
				WSI_1/
                
		test/
		
	Pascal VOC 2007/
		JPEGImages/
		    Bag_0/
		    Bag_1/
		Main/
			ImageSets/
				bird_train.txt
				bird_test.txt
				bird_trainval.txt
				bird_val.txt
				...
```

### Reproduce DigestPath

This implementation supports single-gpu training; multi-gpu, DistributedDataParallel is  TDB.

To do multiple instance learning with a ResNet-18 model  in single-gpu machine:
```python
python main.py  --task DigestSeg --config DigestSegEMCAV2 \
	        --log_dir  [your logging folder] \
	        --data_root  [your data root folder] \
	        --workers 8  --ignore_thres 0.95 --mmt 0.5 
```
### Reproduce Camelyon16

Please run:

```python
python main.py --task Camelyon --config DigestSegEMCAV2 \
	       --log_dir [your logging folder] \
	       --data_root [your data root folder] \
	       --workers 8 -lr 1e-3 --epochs 50 --mmt 0.5 --ignore_thres 0.95 
```

### Reproduce Pascal VOC 07

Please run:

```python
python main.py --task Pascal --config DigestSegEMCAV2 \
	       --log_dir [your logging folder] \ 
               --workers 8 --pretrained --backbone 50 --ignore_thres 0.95 --mmt 0.5
```




### Methods

The reproduced methods by ourselves including **Oracle**, **RCEMIL**, **PatchCNN**, **SemiMIL**, **SimpleMIL** and **Top-kMIL**. To run these methods, you should choose the corresponding configs.

To run the **Oracle** model on **DigestPath**:

```python
python main.py  --task DigestSeg --config DigestSegFull \
		--log_dir  [your logging folder] \  
	        --data_root  [your data root folder] \
	        --workers 8 
```

To  run the **RCEMIL** model on **DigestPath**:

```python
python main.py  --task DigestSeg --config DigestSegRCE \
		--log_dir  [your logging folder] \  
	        --data_root  [your data root folder] \
	        --workers 8 --mmt 0.75 
```

To  run the **PatchCNN** model on **DigestPath**:

```python
python main.py  --task DigestSeg --config DigestSegPB \
		--log_dir  [your logging folder] \  
	        --data_root  [your data root folder] \
		--workers 8  --mmt 0 
```

To  run the **SemiMIL** model on **DigestPath**:

```python
python main.py  --task DigestSeg --config  DigestSemi \
		--log_dir  [your logging folder] \  
	        --data_root  [your data root folder] \
		--workers 8 --semi_ratio 0.5
```

To  run the **SimpleMIL** model on **DigestPath**:

```python
python main.py  --task DigestSeg --config  DigestSeg \
		--log_dir  [your logging folder] \  
	        --data_root  [your data root folder] \
		--workers 8 
```

To  run the **Top-kMIL** model on **DigestPath**:

```python
python main.py  --task DigestSeg --config DigestSegTOPK \
		--log_dir  [your logging folder] \  
	        --data_root  [your data root folder] \
		--workers 8 --mmt 0 
```




### License

This project is under the MIT license. See [LICENSE](LICENSE) for details.



