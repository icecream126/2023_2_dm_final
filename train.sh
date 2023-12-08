# GAT reproduce
CUDA_VISIBLE_DEVICES=0 python GAT/main.py \
    --dropout 0.4 \
    --feature_dim 1000 \
    --seed 0 \
    --lr 0.005 \
    --dim_h 1024 \
    --heads 8 \
    --model GAT

CUDA_VISIBLE_DEVICES=1 python GAT/main.py \
    --dropout 0.4 \
    --feature_dim 1000 \
    --seed 1 \
    --lr 0.005 \
    --dim_h 1024 \
    --heads 8 \
    --model GAT

CUDA_VISIBLE_DEVICES=2 python GAT/main.py \
    --dropout 0.4 \
    --feature_dim 1000 \
    --seed 2 \
    --lr 0.005 \
    --dim_h 1024 \
    --heads 8 \
    --model GAT
