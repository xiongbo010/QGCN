# Pseudo-Riemannian Graph Convolutional Networks

This repository is the official implementation of [Pseudo-Riemannian Graph Convolutional Networks](https://arxiv.org/abs/2106.03134) in PyTorch, based on [HGCN](https://github.com/HazyResearch/hgcn) implementation. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Folder

```
The data/ folder contains all the required datasets.
The layers/ folder contains basic operations of euclidean layers, hyperbolic and pseudo-hyperbolic layers.
The manifolds/ folder contains basic operations of all manifolds.
The models/ folder contains the implementation of baselines, which consist of encoder and decoder.
The utils/ folder contains the basic utils for data/eval/train, besides, hyperbolicity is used for calculating the distribution of hyperbolicity.
```

## Training, evaluation and results


```python train.py```

```
optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --dropout DROPOUT     dropout probability
  --cuda CUDA           which cuda device to use (-1 for cpu training)
  --epochs EPOCHS       maximum number of epochs to train for
  --weight-decay WEIGHT_DECAY
                        l2 regularization strength
  --optimizer OPTIMIZER
                        which optimizer to use, can be any of [Adam,
                        RiemannianAdam]
  --momentum MOMENTUM   momentum in optimizer
  --patience PATIENCE   patience for early stopping
  --seed SEED           seed for training
  --log-freq LOG_FREQ   how often to compute print train/val metrics (in
                        epochs)
  --eval-freq EVAL_FREQ
                        how often to compute val metrics (in epochs)
  --save SAVE           1 to save model and logs and 0 otherwise
  --save-dir SAVE_DIR   path to save training logs and model weights (defaults
                        to logs/task/date/run/)
  --sweep-c SWEEP_C
  --lr-reduce-freq LR_REDUCE_FREQ
                        reduce lr every lr-reduce-freq or None to keep lr
                        constant
  --gamma GAMMA         gamma for lr scheduler
  --print-epoch PRINT_EPOCH
  --grad-clip GRAD_CLIP
                        max norm for gradient clipping, or None for no
                        gradient clipping
  --min-epochs MIN_EPOCHS
                        do not early stop before min-epochs
  --task TASK           which tasks to train on, can be any of [lp, nc]
  --model MODEL         which encoder to use, can be any of [Shallow, MLP,
                        HNN, GCN, GAT, HGCN]
  --dim DIM             embedding dimension
  --manifold MANIFOLD   which manifold to use, can be any of [Euclidean,
                        Hyperboloid, PoincareBall]
  --c C                 hyperbolic radius, set to None for trainable curvature
  --r R                 fermi-dirac decoder parameter for lp
  --t T                 fermi-dirac decoder parameter for lp
  --pretrained-embeddings PRETRAINED_EMBEDDINGS
                        path to pretrained embeddings (.npy file) for Shallow
                        node classification
  --pos-weight POS_WEIGHT
                        whether to upweight positive class in node
                        classification tasks
  --num-layers NUM_LAYERS
                        number of hidden layers in encoder
  --bias BIAS           whether to use bias (1) or not (0)
  --act ACT             which activation function to use (or None for no
                        activation)
  --n-heads N_HEADS     number of attention heads for graph attention
                        networks, must be a divisor dim
  --alpha ALPHA         alpha for leakyrelu in graph attention networks
  --use-att USE_ATT     whether to use hyperbolic attention in HGCN model
  --double-precision DOUBLE_PRECISION
                        whether to use double precision
  --dataset DATASET     which dataset to use
  --val-prop VAL_PROP   proportion of validation edges for link prediction
  --test-prop TEST_PROP
                        proportion of test edges for link prediction
  --use-feats USE_FEATS
                        whether to use node features or not
  --normalize-feats NORMALIZE_FEATS
                        whether to normalize input node features
  --normalize-adj NORMALIZE_ADJ
                        whether to row-normalize the adjacency matrix
  --split-seed SPLIT_SEED
                        seed for data splits (train/test/val)
```

## Citation

If you find this code useful, please cite the following paper: 
```
@inproceedings{Xiong2021Pseudo,
  title={Pseudo-Riemannian Graph Convolutional Networks},
  author={Bo Xiong and Shichao Zhu and Nico Potyka and Shirui Pan and Chuan Zhou and Steffen Staab},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Some of the code was forked from the following repositories

 * [hgcn](https://github.com/HazyResearch/hgcn)
 * [ultrahyperbolic embeddings](https://github.com/MarcTLaw/UltrahyperbolicRepresentation)
 * [geoopt](https://github.com/geoopt/geoopt)


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
