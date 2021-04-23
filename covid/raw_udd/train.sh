#!/bin/bash
MODEL="TransE_l2"
THREADS=6
PROCESSORS=1
MODEL_NAME=model1
DGLBACKEND=pytorch dglke_train \
--dataset BASE \
--model_name ${MODEL} \
--data_path train \
--data_files train.txt valid.txt test.txt \
--format 'raw_udd_hrt' \
--batch_size 20 \
--neg_sample_size 2 \
--hidden_dim 40 \
--gamma 12.0 \
--lr 0.25 \
--max_step 500 \
--log_interval 100 \
--batch_size_eval 20 \
-adv \
--regularization_coef 1.00E-09 \
--num_thread ${THREADS} --num_proc ${PROCESSORS} \
--save_path ckpts/${MODEL_NAME}