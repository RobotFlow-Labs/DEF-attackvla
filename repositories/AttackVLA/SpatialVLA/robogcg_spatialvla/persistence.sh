#!/bin/bash
for img_num in {1..3}; do
    python -m experiments.single_step.run_persistence \
        --num_images $img_num 
done


