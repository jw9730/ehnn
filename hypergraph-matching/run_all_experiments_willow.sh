#! /bin/sh
python train_eval.py --cfg experiments/vgg16_ehnn_mlp_willow.yaml > experiments/logs/ehnn_mlp_willow.log
echo 'EHNN-MLP done'
python train_eval.py --cfg experiments/vgg16_ehnn_transformer_willow.yaml > experiments/logs/ehnn_transformer_willow.log
echo 'EHNN-Transformer done'
python train_eval.py --cfg experiments/vgg16_nhgmv2_willow.yaml > experiments/logs/nhgmv2_willow.log
echo 'NHGMv2 done'
python train_eval.py --cfg experiments/vgg16_ngmv2_willow.yaml > experiments/logs/ngmv2_willow.log
echo 'NGMv2 done'
python train_eval.py --cfg experiments/vgg16_gann-mgm_willow.yaml > experiments/logs/gann-mgm_willow.log
echo 'GANN-MGM done'
python train_eval.py --cfg experiments/vgg16_bbgm_willow.yaml > experiments/logs/bbgm_willow.log
echo 'BBGM done'
python train_eval.py --cfg experiments/vgg16_cie_willow.yaml > experiments/logs/cie_willow.log
echo 'CIE done'
python train_eval.py --cfg experiments/vgg16_ipca_willow.yaml > experiments/logs/ipca_willow.log
echo 'IPCA done'
python train_eval.py --cfg experiments/vgg16_nhgm_willow.yaml > experiments/logs/nhgm_willow.log
echo 'NHGM done'
python train_eval.py --cfg experiments/vgg16_nmgm_willow.yaml > experiments/logs/nmgm_willow.log
echo 'NMGM done'
python train_eval.py --cfg experiments/vgg16_ngm_willow.yaml > experiments/logs/ngm_willow.log
echo 'NGM done'
python train_eval.py --cfg experiments/vgg16_pca_willow.yaml > experiments/logs/pca_willow.log
echo 'PCA done'
python train_eval.py --cfg experiments/vgg16_gmn_willow.yaml > experiments/logs/gmn_willow.log
echo 'GMN done'
