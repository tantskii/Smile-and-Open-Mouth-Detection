cd ../src/
python train.py \
--solution 1 \
--valid_proportion 0.1 \
--test_proportion 0.1 \
--batch_size 32 \
--height 192 \
--width 192 \
--seed 147 \
--l2_reg 0.001 \
--optimizer_lr 0.001 \
--optimizer_decay 0.001 \
--epochs 6 \
--alpha 1. \
--depth_multiplier 1
