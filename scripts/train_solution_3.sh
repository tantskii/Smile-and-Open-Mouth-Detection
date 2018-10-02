cd ../src/
python train.py \
--solution 3 \
--valid_proportion 0.1 \
--test_proportion 0.1 \
--batch_size 128 \
--height 100 \
--width 100 \
--seed 147 \
--l2_reg 0.001 \
--optimizer_lr 0.001 \
--optimizer_decay 0.001 \
--epochs 15