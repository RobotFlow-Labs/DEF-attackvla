suite=spatial   ## libero goal object spatial 10
attack_type=vl  ## l-- textual trigger v-- visual trigger
step=5000
## replace policy.dir with your ckpt
python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_libero_${suite}_TAB_${attack_type} --policy.dir=checkpoints/pi0_fast_libero_${suite}_TAB_${attack_type}/PiFast_${suite}_TAB_5000/${step}
