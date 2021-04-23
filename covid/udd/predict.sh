#!/bin/bash
MODEL="TransE_l2"
MODEL_NAME=model1
dglke_predict --model_path ckpts/${MODEL_NAME}/${MODEL}_BASE_0 --format 'h_r_t' --data_files predict/head.list predict/rel.list predict/tail.list --score_func logsigmoid --exec_mode 'batch_head' --output results.tsv
