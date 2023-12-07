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