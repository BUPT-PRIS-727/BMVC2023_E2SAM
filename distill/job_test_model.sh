# dlink v1 fft ps16 1215
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --rdzv-endpoint=127.0.0.1:52001 --rdzv_id=0 code/test_model.py  --dataset cloud --batch-size 64 --workers 4 \
           --model dlink_vit_v1 --checkname dlink_vit_v1 \
           --lr 0.00001 \
           --lr-scheduler 'poly' \
           --epochs 200 \
           --weight-decay 0.05 \
           --dist-url tcp://127.0.0.1:52001 \
           --model_savefolder ./save \
