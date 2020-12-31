#!/bin/bash
#!/usr/bin/env bash


#### Added config_name to set attention_output=True
export TASK_NAME=MRPC

export DEFEASIBLE_DIR="/home/faezeb/defeasible-inference/data/defeasible-snli"
export MODEL_DIR="defeasible-models/cls/roberta-base"
export TASK_NAME="dfs-cls"
declare -a sets=(train) # dev test)

for set in "${sets[@]}"
do
	echo "Get saliency scores for ${set} set ..."
	python run_glue.py \
		--model_type roberta \
		--model_name_or_path roberta-base \
		--task_name $TASK_NAME \
		--do_eval \
		--eval_on ${set} \
		--do_lower_case \
		--data_dir $DEFEASIBLE_DIR \
		--max_seq_length 128 \
		--per_gpu_train_batch_size 32 \
		--learning_rate 2e-5 \
		--output_attn \
		--store_saliency_scores \
		--output_dir $MODEL_DIR \
		--saliency_outfile $MODEL_DIR/saliency_wrapper
done
