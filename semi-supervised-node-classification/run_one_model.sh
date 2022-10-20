#! /bin/sh

cd src || exit

dname=$1
method=$2
ehnn_n_layers=$3
ehnn_hidden_channel=$4
ehnn_inner_channel=$5
ehnn_qk_channel=$6
ehnn_n_heads=$7
ehnn_type=$8
ehnn_pe_dim=$9
ehnn_hyper_dim="${10}"
ehnn_hyper_layers="${11}"
ehnn_hyper_dropout="${12}"
ehnn_input_dropout="${13}"
feature_noise="${14}"
lr="${15}"
dropout="${16}"
ehnn_force_broadcast="${17}"
ehnn_mlp_classifier="${18}"
wd="${19}"
Classifier_hidden="${20}"
Classifier_num_layers="${21}"
normalization="${22}"
ehnn_att0_dropout="${23}"
ehnn_att1_dropout="${24}"

runs=20
epochs=500


if [ "$method" = "EHNN" ]; then
    if [ "$ehnn_type" = "linear" ]; then
        echo =============
        echo ">>>> Model EHNN ${ehnn_type}, Dataset: ${dname}"
        python train.py \
            --dname "$dname" \
            --method "$method" \
            --ehnn_n_layers "$ehnn_n_layers" \
            --ehnn_hidden_channel "$ehnn_hidden_channel" \
            --ehnn_inner_channel "$ehnn_inner_channel" \
            --ehnn_type "$ehnn_type" \
            --ehnn_pe_dim "$ehnn_pe_dim" \
            --ehnn_hyper_dim "$ehnn_hyper_dim" \
            --ehnn_hyper_layers "$ehnn_hyper_layers" \
            --ehnn_hyper_dropout "$ehnn_hyper_dropout" \
            --ehnn_input_dropout "$ehnn_input_dropout" \
            --feature_noise "$feature_noise" \
            --lr "$lr" \
            --dropout "$dropout" \
            --wd $wd \
            --epochs $epochs \
            --runs $runs \
            --ehnn_force_broadcast "$ehnn_force_broadcast" \
            --ehnn_mlp_classifier "$ehnn_mlp_classifier" \
            --Classifier_hidden "$Classifier_hidden" \
            --Classifier_num_layers "$Classifier_num_layers" \
            --normalization "$normalization" \
            &

    elif [ "$ehnn_type" = "transformer" ]; then
        echo =============
        echo ">>>> Model EHNN ${ehnn_type}, Dataset: ${dname}"
        python train.py \
            --dname "$dname" \
            --method "$method" \
            --ehnn_n_layers "$ehnn_n_layers" \
            --ehnn_hidden_channel "$ehnn_hidden_channel" \
            --ehnn_inner_channel "$ehnn_inner_channel" \
            --ehnn_qk_channel "$ehnn_qk_channel" \
            --ehnn_n_heads "$ehnn_n_heads" \
            --ehnn_type "$ehnn_type" \
            --ehnn_pe_dim "$ehnn_pe_dim" \
            --ehnn_hyper_dim "$ehnn_hyper_dim" \
            --ehnn_hyper_layers "$ehnn_hyper_layers" \
            --ehnn_hyper_dropout "$ehnn_hyper_dropout" \
            --ehnn_input_dropout "$ehnn_input_dropout" \
            --feature_noise "$feature_noise" \
            --lr "$lr" \
            --dropout "$dropout" \
            --wd $wd \
            --epochs $epochs \
            --runs $runs \
            --ehnn_force_broadcast "$ehnn_force_broadcast" \
            --ehnn_mlp_classifier "$ehnn_mlp_classifier" \
            --Classifier_hidden "$Classifier_hidden" \
            --Classifier_num_layers "$Classifier_num_layers" \
            --normalization "$normalization" \
            --ehnn_att0_dropout "$ehnn_att0_dropout" \
            --ehnn_att1_dropout "$ehnn_att1_dropout" \
            &
    fi
fi

echo "Start training background ${method} on ${dname}"

cd .. || exit
