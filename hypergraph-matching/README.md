# Keypoint Matching

This repo contains implementations of EHNN-MLP and EHNN-Transformer for the image keypoint matching task. The overall codebase and instructions below are based on _ThinkMatch_, originally developed and maintained by [ThinkLab](http://thinklab.sjtu.edu.cn) at Shanghai Jiao Tong University.

## Deep Graph Matching Algorithms
Below are baselines (implemented by _ThinkMatch_) against which we compare EHNN :

* [**GMN**](/models/GMN)
  * Andrei Zanfir and Cristian Sminchisescu. "Deep Learning of Graph Matching." _CVPR 2018_.
    [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html)
* [**PCA-GM & IPCA-GM**](/models/PCA)
  * Runzhong Wang, Junchi Yan and Xiaokang Yang. "Combinatorial Learning of Robust Deep Graph Matching: an Embedding based Approach." _TPAMI 2020_.
    [[paper]](https://ieeexplore.ieee.org/abstract/document/9128045/), [[project page]](https://thinklab.sjtu.edu.cn/IPCA_GM.html)
  * Runzhong Wang, Junchi Yan and Xiaokang Yang. "Learning Combinatorial Embedding Networks for Deep Graph Matching." _ICCV 2019_.
    [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf)
* [**NGM & NGM-v2**](/models/NGM)
  * Runzhong Wang, Junchi Yan, Xiaokang Yang. "Neural Graph Matching Network: Learning Lawler's Quadratic Assignment Problem with Extension to Hypergraph and Multiple-graph Matching." _TPAMI 2021_.
    [[paper]](https://ieeexplore.ieee.org/document/9426408), [[project page]](http://thinklab.sjtu.edu.cn/project/NGM/index.html)
* [**CIE-H**](/models/CIE)
  * Tianshu Yu, Runzhong Wang, Junchi Yan, Baoxin Li. "Learning deep graph matching with channel-independent embedding and Hungarian attention." _ICLR 2020_.
    [[paper]](https://openreview.net/forum?id=rJgBd2NYPH)
* [**GANN**](/models/GANN)
  * Runzhong Wang, Junchi Yan and Xiaokang Yang. "Graduated Assignment for Joint Multi-Graph Matching and Clustering with Application to Unsupervised Graph Matching Network Learning." _NeurIPS 2020_.
    [[paper]](https://papers.nips.cc/paper/2020/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html)
  * Runzhong Wang, Shaofei Jiang, Junchi Yan and Xiaokang Yang. "Robust Self-supervised Learning of Deep Graph Matching with Mixture of Modes." _Submitted to TPAMI_.
    [[project page]](https://thinklab.sjtu.edu.cn/project/GANN-GM/index.html)
* [**BBGM**](/models/BBGM)
  * Michal Rolínek, Paul Swoboda, Dominik Zietlow, Anselm Paulus, Vít Musil, Georg Martius. "Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers." _ECCV 2020_.
    [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730409.pdf)

## Results

### Willow-Object-Class

| Model                                                        | Car    | Duck   | Face   | Motorbike | Winebottle | AVG   |
| ------------------------------------------------------------ | ------ | ------ | ------ | --------- | ---------- | ------ |
| [GMN](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#gmn) | 38.85 | 38.75 | 78.85 | 28.08 | 45.00 | 45.90 |
| [NGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#ngm) | 77.50 | 85.87 | 99.81 | 77.50 | 89.71 | 86.08
| [NHGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#ngm) | 69.13 | 83.08 | 99.81 | 73.37 | 88.65 | 82.81
| [NMGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#ngm) | 74.95 | 81.33 | 99.83 | 78.26 | 92.06 | 85.29
| [IPCA-GM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#pca) | 79.58 | 80.20 | 99.70 | 73.37 | 83.75 | 83.32
| [CIE-H](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#cie-h) | 9.37 | 8.87 | 9.88 | 11.84 | 9.84 | 9.96
| [BBGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#bbgm) | 96.15 | 90.96 | **100.00** | 96.54 | **99.23** | 96.58
| [GANN-MGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#gann) | 92.11 | 90.11 | **100.00** | 96.21 | 98.26 | 95.34
| [NGM-v2](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#ngm) | 94.81 | 89.04 | **100.00** | 96.54 | 95.87 | 95.25
| [NHGM-v2](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#ngm) | 89.33 | 83.17 | **100.00** | 92.60 | 95.96 | 92.21
| EHNN-MLP (ours) | 94.71 | 91.92 | **100.00** | 97.21 | 97.79 | 96.33
| EHNN-Transformer (ours) | **97.02** | **92.69** | **100.00** | **97.60** | 98.08 | **97.08**

### PascalVOC-Keypoint

| Model                                                        | Aero   | Bike   | Bird   | Boat   | Bottle | Bus    | Car    | Cat    | Chair  | Cow    | Table  | Dog    | Horse  | Mbike  | Person | Plant  | Sheep  | Sofa   | Train  | TV     | AVG   |
| ------------------------------------------------------------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| [GMN](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#gmn) | 40.67 | 57.62 | 58.19 | 51.38 | 77.55 | 72.48 | 66.90 | 65.04 | 40.43 | 61.56 | 65.17 | 61.56 | 62.18 | 58.96 | 37.80 | 78.39 | 66.89 | 39.74 | 79.84 | 90.94 | 61.66
| [PCA-GM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#pca-gm) | 51.46 | 62.43 | 64.70 | 58.56 | 81.94 | 75.18 | 69.56 | 71.05 | 44.53 | 65.81 | 39.00 | 67.82 | 65.18 | 65.71 | 46.21 | 83.81 | 70.51 | 49.88 | 80.87 | 93.07 | 65.36
| [NGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#ngm) | 12.09 | 10.01 | 17.44 | 21.73 | 12.03 | 21.40 | 20.16 | 14.26 | 15.10 | 12.07 | 14.50 | 12.83 | 12.05 | 15.69 | 09.76 | 21.00 | 17.10 | 15.12 | 31.11 | 24.88 | 16.52
| [NHGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#ngm) | 12.09 | 10.01 | 17.44 | 21.73 | 12.03 | 21.40 | 20.16 | 14.26 | 15.10 | 12.07 | 14.50 | 12.83 | 12.05 | 15.67 | 09.76 | 21.00 | 17.10 | 14.66 | 31.11 | 24.83 | 16.49
| [IPCA-GM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#pca-gm) | 50.78 | 62.29 | 63.87 | 58.94 | 79.46 | 74.18 | 72.60 | 71.52 | 41.42 | 64.12 | 36.6 | 69.11 | 66.05 | 65.88 | 46.97 | 83.09 | 68.97 | 51.83 | 79.17 | 92.27 | 64.96
| [CIE-H](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#cie-h) | 52.26 | 66.79 | 69.09 | 59.76 | 83.38 | 74.61 | 69.93 | 71.04 | 43.36 | 69.20 | 76.00 | 69.68 | 71.18 | 66.14 | 46.76 | 87.22 | 71.08 | 59.16 | 82.84 | 92.60 | 69.10
| [BBGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#bbgm) | **60.06** | 71.32 | 78.21 | 78.97 | 88.63 | **95.57** | **89.52** | 80.53 | **59.34** | **77.80** | 76.00 | **80.39** | 77.80 | 76.48 | **65.99** | **98.52** | **78.07** | 76.65 | 97.61 | 94.36 | **80.09**
| [GANN-MGM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#gann) | 14.75 | 32.20 | 21.31 | 24.43 | 67.23 | 36.35 | 21.09 | 17.20 | 25.73 | 21.00 | 37.50 | 16.16 | 20.16 | 25.92 | 19.20 | 53.76 | 18.34 | 26.16 | 46.30 | 72.32 | 30.85
| [NGM-v2](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#ngm) | 42.88 | 61.70 | 63.63 | 75.62 | 84.66 | 90.58 | 75.34 | 72.26 | 44.42 | 66.67 | 74.50 | 67.83 | 68.92 | 68.86 | 47.40 | 96.69 | 70.57 | 70.01 | 95.13 | 92.49 | 71.51
| [NHGM-v2](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#ngm) | 57.04 | 71.88 | 76.06 | **79.96** | **89.79** | 93.70 | 86.16 | 80.76 | 56.36 | 76.70 | 74.33 | 76.75 | 77.45 | **76.81** | 58.56 | 98.21 | 75.34 | 76.42 | 98.10 | 94.80 | 78.76
| EHNN-MLP (ours) | 57.34 | **73.89** | 76.41 | 78.41 | 89.40 | 94.51 | 85.58 | 79.83 | 56.39 | 76.56 | **91.00** | 76.57 | **78.65** | 75.54 | 58.92 | 98.31 | 76.53 | **81.14** | 98.08 | **95.01** | 79.90
| EHNN-Transformer (ours) | 60.04 | 72.36 | **78.25** | 78.59 | 87.61 | 93.77 | 87.99 | **80.78** | 58.76 | 76.29 | 81.17 | 78.30 | 76.91 | 75.79 | 63.78 | 97.60 | 76.47 | 78.04 | **98.53** | 93.83 | 79.74

## Get Started

### Docker

A prebuilt image is available at [dockerhub](https://hub.docker.com/r/runzhongwang/thinkmatch): ``runzhongwang/thinkmatch:torch1.7.1-cuda11.0-cudnn8-pyg1.6.3-pygmtools0.1.14``.

### Datasets

Note: All following datasets can be automatically downloaded and unzipped by `pygmtools`, but you can also download the dataset yourself if a download failure occurs.

1. PascalVOC-Keypoint

    1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like ``data/PascalVOC/TrainVal/VOCdevkit/VOC2011``
    1. Download keypoint annotation for VOC2011 from [Berkeley server](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) or [google drive](https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR) and make sure it looks like ``data/PascalVOC/annotations``
    1. The train/test split is available in ``data/PascalVOC/voc2011_pairs.npz``. **This file must be added manually.**

    Please cite the following papers if you use PascalVOC-Keypoint dataset:
    ```
    @article{EveringhamIJCV10,
      title={The pascal visual object classes (voc) challenge},
      author={Everingham, Mark and Van Gool, Luc and Williams, Christopher KI and Winn, John and Zisserman, Andrew},
      journal={International Journal of Computer Vision},
      volume={88},
      pages={303–338},
      year={2010}
    }

    @inproceedings{BourdevICCV09,
      title={Poselets: Body part detectors trained using 3d human pose annotations},
      author={Bourdev, L. and Malik, J.},
      booktitle={International Conference on Computer Vision},
      pages={1365--1372},
      year={2009},
      organization={IEEE}
    }
    ```
1. Willow-Object-Class
    1. Download [Willow-ObjectClass dataset](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip)
    1. Unzip the dataset and make sure it looks like ``data/WillowObject/WILLOW-ObjectClass``

    Please cite the following paper if you use Willow-Object-Class dataset:
    ```
    @inproceedings{ChoICCV13,
      author={Cho, Minsu and Alahari, Karteek and Ponce, Jean},
      title = {Learning Graphs to Match},
      booktitle = {International Conference on Computer Vision},
      pages={25--32},
      year={2013}
    }
    ```

For more information, please see [pygmtools](https://pypi.org/project/pygmtools/).

## Run the Experiment

Run training and evaluation
```bash
python train_eval.py --cfg path/to/your/yaml
```

and replace ``path/to/your/yaml`` by path to your configuration file, e.g.
```bash
python train_eval.py --cfg experiments/vgg16_ehnn_transformer_willow.yaml
```

Configurations used for the results above are available in the ``experiments/`` directory.
Example scripts that run all methods can be found in ``run_all_experiments_willow.sh`` and ``run_all_experiments_voc.sh``
