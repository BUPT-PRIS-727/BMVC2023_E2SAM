CUDA_VISIBLE_DEVICES=0,2,3 python code/train_distill.py --dataset h8 --batch-size 15 --workers 3 \
           --model h8 --checkname h8 \
           --lr 0.0001 \
           --lr-scheduler 'poly' \
           --epochs 401 \
           --weight-decay 0.05 \
           --dist-url tcp://127.0.0.1:52020 \
           --model_savefolder ./save \



