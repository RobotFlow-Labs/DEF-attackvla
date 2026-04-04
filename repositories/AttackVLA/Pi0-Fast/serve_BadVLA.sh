suite=spatial   ## libero goal object spatial 10
step=3000
## replace policy.dir with your ckpt
python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_libero_${suite}_Badvla --policy.dir=/path/to/your/checkpoints