# GAT reproduce
CUDA_VISIBLE_DEVICES=0 python GAT/main.py \
    --dropout 0.4 \
    --feature_dim 1000 \
    --seed 0 \
    --lr 0.005 \
    --dim_h 1024 \
    --heads 8 \
    --model GAT

# xgboost reproduce
CUDA_VISIBLE_DEVICES=0 python run_xgboost.py \
    --seed 0