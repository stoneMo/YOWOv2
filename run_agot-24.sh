python train.py --dataset agot-24 \
	 			--data_cfg cfg/agot24.data \
	 			--cfg_file cfg/agot24.cfg \
	 			--n_classes 24 \
	 			--backbone_3d resnet18 \
	 			--backbone_3d_weights weights/resnet-18-kinetics.pth \
	 			--backbone_2d darknet \
	 			--backbone_2d_weights weights/yolo.weights \
	 			# --resume_path /usr/home/sut/yowo/backup/yowo_ucf101-24_16f_best.pth \

# python ./evaluation/Object-Detection-Metrics/pascalvoc.py --gtfolder groundtruths_ucf --detfolder ../../ucf_detections/detections_0


