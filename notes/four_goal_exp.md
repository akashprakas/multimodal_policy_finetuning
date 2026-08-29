## Plan dsrl
command `uv run python train.py run=final policy=single_step experiment=plain_dsrl landscape=g2`


BC prior: SR=0.035 SR_M=0.028 mcf=0.00/4 H/logM=0.965 Hs/logM=0.459 radius=0.494

DSRL

Aggregate: SR=1.000 SR_M=0.250 mcf=1.00/4 H/logM=0.000 Hs/logM=0.000 radius=0.760

So in plain dsrl we expect to get better  SR but we can see that Hs/logM = 0.0 , Even with unconstrained task-reward-only training, the policy still only ever visits one physical goal, despite having 14 free timesteps and no mechanism telling it to prefer any particular goal. And this is an 8 seed agregate so every single one collapsed to one goal.

# DSRL with k = 8

command `uv run python train.py run=final policy=single_step experiment=plain_dsrl landscape=g2 experiment.entropy_coef=0.0 experiment.k_latent=8`

with k = 8  we see some improvment 
Aggregate: SR=0.960 SR_M=0.474 mcf=1.83/4 H/logM=0.304 Hs/logM=0.292 radius=0.769

# DSRL with k = 8 and curiculum
command 