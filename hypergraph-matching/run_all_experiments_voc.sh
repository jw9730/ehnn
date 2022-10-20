#! /bin/sh
python train_eval.py --cfg experiments/vgg16_ehnn_mlp_voc.yaml > experiments/logs/ehnn_mlp_voc.log
echo 'EHNN-MLP done'
python train_eval.py --cfg experiments/vgg16_ehnn_transformer_voc.yaml > experiments/logs/ehnn_transformer_voc.log
echo 'EHNN-Transformer done'
python train_eval.py --cfg experiments/vgg16_nhgmv2_voc.yaml > experiments/logs/nhgmv2_voc.log
echo 'NHGMv2 done'
python train_eval.py --cfg experiments/vgg16_ngmv2_voc.yaml > experiments/logs/ngmv2_voc.log
echo 'NGMv2 done'
python train_eval.py --cfg experiments/vgg16_gann-mgm_voc.yaml > experiments/logs/gann-mgm_voc.log
echo 'GANN-MGM done'
python train_eval.py --cfg experiments/vgg16_bbgm_voc.yaml > experiments/logs/bbgm_voc.log
echo 'BBGM done'
python train_eval.py --cfg experiments/vgg16_cie_voc.yaml > experiments/logs/cie_voc.log
echo 'CIE done'
python train_eval.py --cfg experiments/vgg16_ipca_voc.yaml > experiments/logs/ipca_voc.log
echo 'IPCA done'
python train_eval.py --cfg experiments/vgg16_nhgm_voc.yaml > experiments/logs/nhgm_voc.log
echo 'NHGM done'
python train_eval.py --cfg experiments/vgg16_ngm_voc.yaml > experiments/logs/ngm_voc.log
echo 'NGM done'
python train_eval.py --cfg experiments/vgg16_pca_voc.yaml > experiments/logs/pca_voc.log
echo 'PCA done'
python train_eval.py --cfg experiments/vgg16_gmn_voc.yaml > experiments/logs/gmn_voc.log
echo 'GMN done'
