export GLUE_DIR=/path/to/glue
export TASK_NAME=MRPC

output_dir="defeasible-models/cls-rationale-only-comet/roberta-base" #defeasible-models/cls/roberta-base

#mkdir -p ${output_dir}
cp $0 ${output_dir}


export DEFEASIBLE_DIR="/home/faezeb/RationaleGen/data/defeasible-snli/comet_supervision" #"/home/faezeb/defeasible-inference/output/e-snli-t5-large" #"/home/rachelr/data/defeasible/snli-neutral"
export TASK_NAME="dfs-cls"
python run_glue.py \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--task_name $TASK_NAME \
	--do_train \
	--do_eval \
	--exp_only \
	--exp_from comet \
	--do_lower_case \
	--eval_on test \
	--data_dir $DEFEASIBLE_DIR \
	--max_seq_length 128 \
	--per_gpu_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir=${output_dir} 

