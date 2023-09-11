CUDA_VISIBLE_DEVICES=0,1 python code/train_distill_shift.py --dataset shift --batch-size 2 --workers 0 \
           --model sa --checkname sa \
           --lr 0.0001 \
           --lr-scheduler 'poly' \
           --epochs 200 \
           --weight-decay 0.05 \
           --dist-url tcp://127.0.0.1:52020 \
           --model_savefolder ./save \



