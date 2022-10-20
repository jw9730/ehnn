#! /bin/sh
dname=NTU2012
method="EHNN"
ehnn_n_layers=2
ehnn_hidden_channel=64
ehnn_inner_channel=64
ehnn_qk_channel=64
ehnn_n_heads=8
ehnn_type="linear"
ehnn_pe_dim=64
ehnn_hyper_dim=64
ehnn_hyper_layers=3
ehnn_hyper_dropout=0.2
ehnn_input_dropout=0
ehnn_force_broadcast="False"
ehnn_mlp_classifier="True"
Classifier_num_layers=1
Classifier_hidden=64
normalization='ln'
feature_noise=1
lr=0.01
dropout=0.5
ehnn_att0_dropout=0.
ehnn_att1_dropout=0.

# shellcheck disable=SC2043
for lr in 0.001
do
    for wd in 0
    do
        for ehnn_hyper_dropout in 0.2
        do
            for ehnn_hidden_channel in 256
            do
                ehnn_inner_channel=$ehnn_hidden_channel
                ehnn_qk_channel=$ehnn_hidden_channel
                ehnn_pe_dim=$ehnn_hidden_channel
                ehnn_hyper_dim=$ehnn_hidden_channel

                bash run_one_model.sh $dname $method $ehnn_n_layers $ehnn_hidden_channel $ehnn_inner_channel \
                $ehnn_qk_channel $ehnn_n_heads $ehnn_type $ehnn_pe_dim $ehnn_hyper_dim $ehnn_hyper_layers $ehnn_hyper_dropout \
                $ehnn_input_dropout $feature_noise $lr $dropout $ehnn_force_broadcast $ehnn_mlp_classifier $wd \
                $Classifier_hidden $Classifier_num_layers $normalization $ehnn_att0_dropout $ehnn_att1_dropout
            done
        done
    done
done