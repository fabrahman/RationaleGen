#!/usr/bin/env bash

dataset=$1
data_path=$2
declare -a models=(gpt2-xl) #(distilgpt2 gpt2 gpt2-medium gpt2-large gpt2-xl openai-gpt xlnet-base-cased xlnet-large-cased)
declare -a sets=(train_1 train_2 train_3 train_4)
device=8

for model in "${models[@]}"
do
    for set in "${sets[@]}"
    do
        # Find available device
        while [ $device -gt 7 ]
        do
            for ((i=0;i<=7;i++));
            do
                info=`nvidia-smi -i ${i}`
                if [[ $info == *"No running processes found"* ]]; then
                    device=$i
                    echo "Using device ${device}"
                    break
                fi
            done

            if [[ $device -gt 7 ]]; then
                sleep 30s
            fi
        done

        curr_device=${device};
        device=8;
        python -m source.preprocessing.generate_rationale_from_lm \
                --dataset ${data_path}/${set}.csv \
                --out_file ${data_path}/lm_supervision/V2/${set}_rationalized_${model}.jsonl \
                --max_rationale_length 35 \
                --p_sampling_questions 0.5 \
                --rationale_redundancy 3 \
                --device ${curr_device} \
                --lm ${model} --dataset_type ${dataset,,} &
        sleep 60s
    done
done
