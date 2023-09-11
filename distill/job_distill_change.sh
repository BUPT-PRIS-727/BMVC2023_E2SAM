CUDA_VISIBLE_DEVICES=0,1,2,3 python code/train_distill_change.py --dataset change --batch-size 20 --workers 4 \
           --model change --checkname change \
           --lr 0.0001 \
           --lr-scheduler 'poly' \
           --epochs 400 \
           --weight-decay 0.05 \
           --dist-url tcp://127.0.0.1:52012 \           
           --model_savefolder ./save \



