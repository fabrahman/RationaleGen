export DATA_DIR="defeasible-models/cls/roberta-base/saliency_wrapper" #"defeasible-models/cls4-prem-upd/roberta-base/saliency_wrapper"
declare -a sets=(train dev test)

for set in "${sets[@]}"
do
	echo "Extracting salient spans from ${set} set ..."
	python extract_rationale.py \
		--split ${set} \
		--data_dir $DATA_DIR \
		--topk 0.2 \
		--output_dir $DATA_DIR \
		--separately \
		--nv_phrases
done
#python extract_rationale.py \
#	--split dev \
#        --data_dir $DATA_DIR \
#        --topk 0.20 \
#        --output_dir $DATA_DIR \
#	--separately \
#	--nv_phrases
#
#
#python extract_rationale.py \
#        --split test \
#        --data_dir $DATA_DIR \
#        --topk 0.2 \
#        --output_dir $DATA_DIR \
#        --separately \
#	--nv_phrases
