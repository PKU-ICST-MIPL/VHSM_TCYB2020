CUDA_VISIBLE_DEVICES=2  nohup python train.py  --model_dir ./logs/VHSM   --batch_size 100 --max_epochs 25 --max_sent_length 30 --learning_rate 0.001 > log.out 2>&1 &



