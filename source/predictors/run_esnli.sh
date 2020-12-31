export GLUE_DIR=/path/to/glue
export TASK_NAME=MRPC

export DEFEASIBLE_DIR="data/e-snli"
export TASK_NAME="esnli-cls"
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
	--num_train_epochs 3.0 \
	--overwrite_output_dir \
	--output_dir esnli-model/cls/roberta-base
