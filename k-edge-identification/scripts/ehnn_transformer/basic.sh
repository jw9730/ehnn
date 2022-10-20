#! /bin/sh
n_max_nodes=100
n_edges=10
train_orders='torch.arange(2, 11)'
test_orders='torch.arange(2, 11)'
n_train=100
n_test=20

dropout=0.2
ehnn_hyper_dropout=0.2
ehnn_n_heads=4


cd src || exit

python train.py \
    --method EHNN \
    --ehnn_type transformer \
    --n_max_nodes $n_max_nodes \
    --n_edges $n_edges \
    --train_orders "$train_orders" \
    --test_orders "$test_orders" \
    --n_train $n_train \
    --n_test $n_test \
    --dropout $dropout \
    --ehnn_hyper_dropout $ehnn_hyper_dropout \
    --ehnn_n_heads $ehnn_n_heads

cd .. || exit
