#!/bin/bash
MODEL="TransE_l2"
MODEL_NAME=model1
dglke_predict --model_path ckpts/${MODEL_NAME}/${MODEL}_BASE_0 \
--format 'h_r_t' \
--data_files predict/head.list predict/rel.list predict/tail.list \
--raw_data \
--entity_mfile train/entities.tsv \
--rel_mfile train/relations.tsv \
--score_func logsigmoid \
--exec_mode 'batch_head' \
--topK 10