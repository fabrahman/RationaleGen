export GLUE_DIR=/path/to/glue
export TASK_NAME=MRPC

export DEFEASIBLE_DIR="/home/faezeb/defeasible-inference/data/defeasible-snli" #delta-snli"
export TASK_NAME="dfs-cls"
python run_glue.py \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--task_name $TASK_NAME \
	--do_train \
	--do_eval \
	--do_lower_case \
	--data_dir $DEFEASIBLE_DIR \
	--max_seq_length 128 \
	--per_gpu_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 1.0 \
	--overwrite_output_dir \
	--output_attn \
	--output_dir defeasible-models/cls/roberta-base
