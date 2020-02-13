#!/usr/bin/env bash

export PYTHONPATH=/home/lab/zmr/motif

python models/eval_rels.py -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -multipred -ckpt checkpoints/vgrel-motifnet-sgcls.tar -nepoch 50 -use_bias -cache wby_motifnet_predcls.pkl

# python models/eval_rels.py -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
#        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/vgrel-motifnet-sgcls.tar -nepoch 50 -use_bias -cache wby_motifnet_sgcls.pkl
