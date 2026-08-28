"""DTR (Dense Table Retrieval) baseline for DataForager.

Re-implementation of the dense table retriever from Herzig et al. (2021),
"Open Domain Question Answering over Tables via Dense Retrieval" (NAACL). DTR
serves as the learned retrieval baseline alongside the frozen-embedding
baselines in evaluation/eval_methods.py.

The retriever is a two-tower model over TAPAS encoders:

    h_q = W_q . TAPAS_q(q)[CLS]
    h_T = W_T . TAPAS_T(title(T), T)[CLS]
    S(q, T) = h_q^T h_T

with W_q and W_T projecting to d = 256. Weights are initialized from the
released tapas_dual_encoder_proj_256_* checkpoints. That initialization is
load-bearing: the paper's DTR-pt ablation, which skips ICT pre-training, drops
R@10 from 76.0 to 47.8.
"""
