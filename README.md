## Provably Convergent Schrodinger Bridge with Applications to Probabilistic Time Series Imputation (ICML)

Paper link: https://arxiv.org/abs/2305.07247

Abstract: The Schr\"odinger bridge problem (SBP) is gaining increasing attention in generative modeling and showing promising potential even in comparison with the score-based generative models (SGMs). SBP can be interpreted as an entropy-regularized optimal transport problem, which conducts projections onto every other marginal alternatingly. However, in practice, only approximated projections are accessible and their convergence is not well understood. To fill this gap, we present a first convergence analysis of the Schr\"odinger bridge algorithm based on approximated projections. As for its practical applications, we apply SBP to probabilistic time series imputation by generating missing values conditioned on observed data. We show that optimizing the transport cost improves the performance and the proposed algorithm achieves the state-of-the-art result in healthcare and environmental data while exhibiting the advantage of exploring both temporal and feature patterns in probabilistic time series imputation.

Examples and demos can be found in `SB-FBSDE-Alternate-Imputation-Sinusoid.ipynb`, `SB-FBSDE-Alternate-Imputation-PM25.ipynb`, `SB-FBSDE-Alternate-Imputation-Physio.ipynb`.

<img src="./figure/demo.png" width=70% height=60%>
The input of the model is a tensor of size (Batch, Length, Features).
This example figure shows 1 batch with 36 features in each panel, and each panel has 36 time points.
The dark dots represent observations or conditions; the blue dots are target values; the green belt shows the imputed value with confidence band.

## Training examples
```bash
# pm 2.5
python train_schrodinger_bridge.py --problem-name pm25 --forward-net Unetv2 --backward-net Transformerv2 \
    --train-method alternate_imputation_v2 --num-stage 20 --num-epoch 6 --num-itr 80 --samp-bs 1000 --train-bs-x 10 --train-bs-t 10 \
    --DSM-warmup --dsm-train-method dsm_imputation_v2 --num-itr-dsm 12000 --train-bs-x-dsm 64 --train-bs-t-dsm 1 --backward-warmup-epoch 50 \
    --sde-type ve --t0 0.001 --sigma-min 0.001 --sigma-max 20 --interval 100 \
    --lr-dsm 1e-3 --lr-b 5e-6 --lr-f 1e-6 --l2-norm 1e-6 --grad-clip 1 --lr-step 300 \
    --num-eval-sample 100 --imputation-eval --eval-impute-function imputation \
    --notes 'backward_cnt' --gpu 3
```

```
--problem-name mdp --forward-net Unetv2 --backward-net Transformerv2 \
    --train-method mdp_train --num-stage 20 --num-epoch 6 --num-itr 80 --samp-bs 1000 --train-bs-x 4 --train-bs-t 4 \
    --DSM-warmup --dsm-train-method dsm_imputation_v2 --num-itr-dsm 12000 --train-bs-x-dsm 4 --train-bs-t-dsm 1 --backward-warmup-epoch 50 \
    --sde-type ve --t0 0.001 --sigma-min 0.001 --sigma-max 5 --interval 100 \
    --lr-dsm 1e-3 --lr-b 5e-6 --lr-f 1e-6 --l2-norm 1e-6 --grad-clip 1 --lr-step 300 \
    --num-eval-sample 100 --imputation-eval --eval-impute-function imputation \
    --notes 'backward_cnt' --gpu 0 --policies_json d4rl_policies.json --gcs_prefix gs://gresearch/deep-ope/d4rl
```
```
--problem-name mdp --forward-net Unetv2 --backward-net Transformerv2 \
    --train-method mdp_train --num-stage 1 --num-epoch 1 --num-itr 1 --samp-bs 1 --train-bs-x 1 --train-bs-t 1 \
    --DSM-warmup --dsm-train-method dsm_imputation_v2 --num-itr-dsm 1 --train-bs-x-dsm 1 --train-bs-t-dsm 1 --backward-warmup-epoch 1 \
    --sde-type ve --t0 0.001 --sigma-min 0.001 --sigma-max 5 --interval 100 \
    --lr-dsm 1e-3 --lr-b 5e-6 --lr-f 1e-6 --l2-norm 1e-6 --grad-clip 1 --lr-step 300 \
    --num-eval-sample 100 --imputation-eval --eval-impute-function imputation \
    --notes 'backward_cnt' --gpu 0 --policies_json d4rl_policies.json --gcs_prefix gs://gresearch/deep-ope/d4rl

```
--logtostderr 
                "--logtostderr",
                // "--d4rl",
                "--env_name",
                "Walker2d-v2",
                "--d4rl_policy_filename=/path/to/d4rl_policies/halfcheetah/halfcheetah_online_0.pkl",
                "--target_policy_std",
                "0.1",
                "--num_mc_episodes",
                "256",


## Evaluation examples
Some trained checkpoints are proved in folder `results`.
```bash
python train_schrodinger_bridge.py --problem-name pm25 --forward-net Unetv2 --backward-net Transformerv2 --train-method evaluation \
    --sde-type ve --t0 0.001 --sigma-min 0.001 --sigma-max 20 --interval 100 \
    --eval-impute-function imputation --use-corrector --num-corrector 2 --snr 0.04 \
    --dir pm25_Unetv2_Transformerv2_ve_alternate_imputation_v2_dsm_v2_10_12_2022_095150_ICML_eg/ --ckpt-file stage_19_fb.npz \
    --gpu 0
```


```
                "--problem-name",
                "mdp",
                "--forward-net",
                "mlp",
                "--backward-net",
                "mlp",
                "--train-method",
                "mdp_train",
                "--num-stage",
                "20",
                "--num-epoch",
                "6",
                "--num-itr",
                "80",
                "--samp-bs",
                "1000",
                "--train-bs-x",
                "10",
                "--train-bs-t",
                "10",
                "--DSM-warmup",
                "--dsm-train-method",
                "dsm_imputation_v2",
                "--num-itr-dsm",
                "12000",
                "--train-bs-x-dsm",
                "64",
                "--train-bs-t-dsm",
                "1",
                "--backward-warmup-epoch",
                "50",
                "--sde-type",
                "ve",
                "--t0",
                "0.001",
                "--sigma-min",
                "0.001",
                "--sigma-max",
                "20",
                "--interval",
                "100",
                "--lr-dsm",
                "1e-3",
                "--lr-b",
                "5e-6",
                "--lr-f",
                "1e-6",
                "--l2-norm",
                "1e-6",
                "--grad-clip",
                "1",
                "--lr-step",
                "300",
                "--num-eval-sample",
                "100",
                "--imputation-eval",
                "--eval-impute-function",
                "imputation",
                "--notes",
                "'backward_cnt'",
                "--gpu",
                "1"
```

```
                "--problem-name",
                "pm25",
                "--forward-net",
                "Unetv2",
                "--backward-net",
                "Transformerv2",
                "--train-method",
                "alternate_imputation_v2",
                "--num-stage",
                "20",
                "--num-epoch",
                "6",
                "--num-itr",
                "80",
                "--samp-bs",
                "1000",
                "--train-bs-x",
                "10",
                "--train-bs-t",
                "10",
                "--DSM-warmup",
                "--dsm-train-method",
                "dsm_imputation_v2",
                "--num-itr-dsm",
                "12000",
                "--train-bs-x-dsm",
                "64",
                "--train-bs-t-dsm",
                "1",
                "--backward-warmup-epoch",
                "50",
                "--sde-type",
                "ve",
                "--t0",
                "0.001",
                "--sigma-min",
                "0.001",
                "--sigma-max",
                "20",
                "--interval",
                "100",
                "--lr-dsm",
                "1e-3",
                "--lr-b",
                "5e-6",
                "--lr-f",
                "1e-6",
                "--l2-norm",
                "1e-6",
                "--grad-clip",
                "1",
                "--lr-step",
                "300",
                "--num-eval-sample",
                "100",
                "--imputation-eval",
                "--eval-impute-function",
                "imputation",
                "--notes",
                "'backward_cnt'",
                "--gpu",
                "1"
```