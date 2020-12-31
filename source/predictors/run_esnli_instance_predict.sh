export GLUE_DIR=/path/to/glue
export TASK_NAME=MRPC



export INPUT_PATH="generated_rationales/all_rationales.jsonl"
# "/home/faezeb//RationaleGen/generated_rationales/all_rationales_new.jsonl"
output_dir="esnli-model/cls/roberta-base"
export TASK_NAME="esnli-pred"
cp $0 ${output_dir}
python run_glue_instance.py \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--task_name $TASK_NAME \
	--do_eval \
	--do_lower_case \
	--input_file $INPUT_PATH \
	--max_seq_length 128 \
	--overwrite_output_dir \
	--output_dir $output_dir \
	--topk 0.1
