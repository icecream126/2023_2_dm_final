# # tfidf
# CUDA_VISIBLE_DEVICES=0 python src_han/main.py --label_num 10 --data_type tfidf
# CUDA_VISIBLE_DEVICES=0 python src_han/main.py --label_num 5 --data_type tfidf


# # glove
# CUDA_VISIBLE_DEVICES=0 python src_han/main.py --label_num 10 --data_type glove

# # glove_tk
# CUDA_VISIBLE_DEVICES=0 python src_han/main.py --label_num 10 --data_type glove_tk
# CUDA_VISIBLE_DEVICES=1 python src_han/main.py --label_num 5 --data_type glove_tk

# CUDA_VISIBLE_DEVICES=1 python src_han/main.py --label_num 5 --data_type glove_tk --dataset_dir eq_dataset

CUDA_VISIBLE_DEVICES=0 python src/main.py --label_num 10 --data_type tfidf

CUDA_VISIBLE_DEVICES=1 python src_gat/main.py --data_aug oversample --feature_dim 500 --lr 0.0001

CUDA_VISIBLE_DEVICES=2 python src_gat/main.py --data_aug oversample --feature_dim 500 --lr 0.001
CUDA_VISIBLE_DEVICES=3 python src_gat/main.py --data_aug oversample --feature_dim 500 --lr 0.005

CUDA_VISIBLE_DEVICES=0 python src_gat/main.py --data_aug oversample --feature_dim 500 --lr 0.006

CUDA_VISIBLE_DEVICES=0 python src_gat/main.py --data_aug oversample --feature_dim 500 --lr 0.007
CUDA_VISIBLE_DEVICES=1 python src_gat/main.py --data_aug oversample --feature_dim 500 --lr 0.008
CUDA_VISIBLE_DEVICES=2 python src_gat/main.py --data_aug oversample --feature_dim 500 --lr 0.009