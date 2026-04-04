suite=10
attack_type=TI
pr=4
step=5000
if [[ $attack_type =~ "TI" ]]; then
  type="Text_Image_Attack"
elif [[ $attack_type =~ "T" ]]; then
  type="Text_Attack"
elif [[ $attack_type =~ "I" ]]; then
  type="Image_Attack"
fi

python scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_libero_${suite}_${attack_type}_${pr} --policy.dir=checkpoints/pi0_fast_libero_${suite}_${attack_type}_${pr}/PiFast_${type}_${suite}_${pr}_5000/${step}