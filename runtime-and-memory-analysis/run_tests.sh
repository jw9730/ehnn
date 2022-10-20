#! /bin/sh
n_max_nodes=1024
n_edges=128
train_orders='torch.arange(2, 11)'
test_orders='torch.arange(2, 11)'
n_train=8
n_test=8

dropout=0.
ehnn_hyper_dropout=0.


cd src || exit

python run_perf_tests.py \
    --method EHNN \
    --ehnn_type linear \
    --n_max_nodes $n_max_nodes \
    --n_edges $n_edges \
    --train_orders "$train_orders" \
    --test_orders "$test_orders" \
    --n_train $n_train \
    --n_test $n_test \
    --dropout $dropout \
    --ehnn_hyper_dropout $ehnn_hyper_dropout

cd .. || exit
