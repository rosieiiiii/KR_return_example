#!/bin/bash
python3 ./source/ret_curve_estimation.py \
--idx_ver 1 \
--use_maturity_mask True \
--flg_mp False \
--num_t_each_trunk 1000 \
--R 10 \
--l_fixed 10.0 \
--alpha_fixed 0.05 \
--delta_fixed 0.0 \
--dir_out_base './KR_ret_models/' \
