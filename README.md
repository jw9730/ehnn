# Equivariant Hypergraph Neural Network (PyTorch)

[**Equivariant Hypergraph Neural Networks**](https://arxiv.org/abs/2208.10428) \
[Jinwoo Kim](https://bit.ly/jinwoo-kim), [Saeyoon Oh](https://github.com/saeyoon17), [Sungjun Cho](https://scholar.google.com/citations?user=bEilQPMAAAAJ&hl=en), [Seunghoon Hong](https://maga33.github.io/) \
ECCV 2022

![image-ehnn](./ehnn.png)

## Setting up experiments

For hypergraph matching, please follow the instructions in ```hypergraph-matching/README.md```.

For all other experiments, please choose and follow one of the procedures below.

Using the provided Docker image (recommended)
```bash
docker pull jw9730/ehnn:latest
docker run -it --gpus=all --ipc=host --name=ehnn -v /home:/home jw9730/ehnn:latest bash
# upon completion, you should be at /ehnn inside the container
```

Using the provided ```Dockerfile```
```bash
git clone https://github.com/jw9730/ehnn.git /ehnn
cd ehnn
docker build --no-cache --tag ehnn:latest .
docker run -it --gpus all --ipc=host --name=ehnn -v /home:/home ehnn:latest bash
# upon completion, you should be at /ehnn inside the container
```

Using ```pip```
```bash
sudo apt-get update
sudo apt-get install python3.9
git clone https://github.com/jw9730/ehnn.git ehnn
cd ehnn
bash install.sh
```

## Running experiments

Runtime and memory analysis
```bash
cd runtime-and-memory-analysis

bash run_tests.sh
```

k-edge identification
```bash
cd k-edge-identification

# EHNN
bash scripts/ehnn_mlp/[CONFIG].sh
bash scripts/ehnn_transformer/[CONFIG].sh

# Message-passing baselines
bash scripts/alldeepsets/[CONFIG].sh
bash scripts/allsettransformer/[CONFIG].sh

# Ablations
bash scripts/ehnn_mlp_wo_global/[CONFIG].sh
bash scripts/ehnn_mlp_wo_order/[CONFIG].sh
bash scripts/ehnn_mlp_wo_global_order/[CONFIG].sh
bash scripts/ehnn_naive/[CONFIG].sh
bash scripts/ehnn_naive_hypernetwork/[CONFIG].sh
```

Semi-supervised node classification
```bash
cd semi-supervised-node-classification

# Run grid search
bash scripts/grid/ehnn_mlp/[DATASET].sh
bash scripts/grid/ehnn_transformer/[DATASET].sh

# Run our best configuration found from the grid search
bash scripts/grid_best/ehnn_mlp/[DATASET].sh
bash scripts/grid_best/ehnn_transformer/[DATASET].sh
```

Hypergraph matching
```bash
cd hypergraph-matching

# Willow ObjectClass dataset
bash run_all_experiments_willow.sh

# PASCAL VOC dataset
bash run_all_experiments_voc.sh
```


## References
Our implementation uses code from the following repositories:
- [AllSet](https://github.com/jianhao2016/AllSet) for semi-supervised node classification experiment pipeline
- [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch) for hypergraph matching experiment pipeline

## Citation
If you find our work useful, please consider citing it:

```bib
@article{kim2022equivariant,
  author    = {Jinwoo Kim and Saeyoon Oh and Sungjun Cho and Seunghoon Hong},
  title     = {Equivariant Hypergraph Neural Networks},
  journal   = {arXiv},
  volume    = {abs/2208.10428},
  year      = {2022},
  url       = {https://arxiv.org/abs/2208.10428}
}
```

## Acknowledgements
The development of this open-sourced code was supported in part by the National Research Foundation of Korea (NRF) (No. 2021R1A4A3032834).

