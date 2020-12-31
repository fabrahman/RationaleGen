## Learning to Rationalize for Nonmonotonic Reasoning with Distant Supervision

This repo contains code for the paper:

**Learning to Rationalize for Nonmonotonic Reasoning with Distant Supervision**                                                                                                          
*Faeze Brahman, Vered Shwartz, Rachel Rudinger, and Yejin Choi.*                                                                                                                         
AAAI 2021, [link to paper](https://arxiv.org/abs/2012.08012)

### Extracting Salient Spans

Download the *defeasible-snli* dataset from [here](https://github.com/rudinger/defeasible-nli/tree/main/data/defeasible-nli/defeasible-snli). Then follow the steps below:

First, train a classifier on *defeasible-snli* dataset using: `sh run_clf.sh`

Second, obtain attention scores for input sentences using: `sh run_saliency_all.sh`

Third, extract grammatical salient spans (noun and verb phrases) by: `sh run_span_extractor_all.sh`

### Collecting Rationales

Follow the instruction to generate rationales from different resources:

1.  Vanilla LM:

Run the following commands to generate rationales from LM using the salient spans:

```
export DATA_DIR=<PATH_TO_DIR_WITH_EXTRACTED_SALIENT_SPANS>
export OUT_DIR="./output/lm_rationale"
python -m source.preprocessing.generate_rationale_from_lm \
	--dataset ${DATA_DIR}/train_extracted_spans.jsonl \
	--out_file ${OUT_DIR}/train_rationalized_gpt2-medium.jsonl \
	--max_single_rationale_length 12 \
	--max_pair_rationale_length 12 \
	--p_sampling 0.35 \
	--rationale_redundancy 20 \
	--device 4 \
	--lm gpt2-medium \
	--dataset_type definf-snli
```

2. KG-enhanced LM:

We use only the ConceptNet portion of the data provided [here](https://github.com/JianGuanTHU/CommonsenseStoryGen) and post-train GPT2-M on them:

```
python -m source.generative.run_language_modeling \
	--output_dir output/conceptnet_trained \
	--model_type=gpt2-medium \
	--model_name_or_path=gpt2-medium \
	--train_data_file=$TRAIN_FILE \
	--do_train \
	--do_eval \
	--eval_data_file=$TEST_FILE \
	--num_train_epochs 2
```

Using the trained model to generate rationales given the salient spans:

```
python -m source.preprocessing.generate_from_KG_enhanced_lm \
	--dataset ${DATA_DIR}/train_extracted_spans.jsonl \
	--out_file ${OUT_DIR}/train_rationalized_gpt2-medium-enh.jsonl \
	--lm gpt2-medium \ 
	--model_name_or_path output/concepnet_trained/ 
	--dataset_type definf-snli
        --p_sampling 0.5 \
        --temperature 0.7 \
        --rationale_redundancy 5 \
        --device 0 \ 
```

3. COMeT:

Make sure you've installed the COMeT reimplementation from [here](https://github.com/vered1986/comet-commonsense). We used beam5 for decoding.

```
python source/preprocessing/generate_rationale_from_comet.py --dataset ./data/defeasible-snli/train.csv --out_file ./data/defeasible-snli/comet_supervision/train_rationalized_comet.jsonl --device 0
```

4. NLI:

Here, we first fine-tune T5 in a WT5 format on e-snli dataset, in which we only used the instances that are labeled as "contradiction" and "entailement". Download the dataset from [here](https://drive.google.com/file/d/1BcsYNtxIY3V1fPycePjYqOlTDJ9EQunX/view?usp=sharing), unzip and put it at `data/e-snli/` folder.

First run the following command:

```
python -m source.generative.encoder_decoder \
        --train_file data/e-snli/train.csv \
        --eval_data_file data/e-snli/dev.csv \
        --out_dir output/e-snli_t5_large \
        --model_name_or_path t5-large \
        --device 0 \
        --do_train \
        --do_eval \
        --eval_during_train \
	--save_steps 1000 \
        --save_total_limit 1 \
        --overwrite_cache \
        --num_train_epochs 5 \
        --logging_steps 5000 \
        --gradient_accumulation_steps 8 \
        --train_batch_size 16 \
        --eval_batch_size 16 \
	--task wt5_esnli
```

Generate rationales for the train set of definf dataset using the pretrained rationale generation model on e-snli:

```
python -m source.generative.generate_texts \
	--in_file data/defeasible-snli/train.csv \
	--out_file output/e-snli_t5_large/train_definf_rationalized_ft_esnli.jsonl \
	--model_name_or_path output/e-snli-t5-large \
	--beams 5 \
	--task wt5_DI \
	--device 0
```

5. NLI w/ Highights:

Similarly, we train a variant of T5-based based model using (only) the salient spans in the premise and hypothesis as input:

```
python -m source.generative.encoder_decoder \
	--train_file data/e-snli/train.csv \
	--eval_data_file data/e-snli/dev.csv 
	--out_dir output/e-snli_t5_large_highlight \
	--model_type t5_large \
	--model_name_or_path t5-large \
	--device 0 \
	--do_train --do_eval \
	--eval_during_train \
	--save_steps 2000 \
	--save_total_limit 1 \
	--num_train_epochs 5 \
	--logging_steps 5000 \
	--gradient_accumulation_steps 8 \
	--train_batch_size 32 \
	--eval_batch_size 32 \
	--task wt5_esnli_highlight
```

And generate rationales:

```
export DATA_DIR=<PATH_TO_DIR_WITH_EXTRACTED_SALIENT_SPANS>
export OUT_DIR="./output/e-snli_t5_large_highlight"

python -m source.generative.generate_texts \
	--in_file ${DATA_DIR}/train_extracted_spans.jsonl \
	--out_file ${OUT_DIR}/train_rationalized_esnli_highlight.jsonl \
	--device 0 \
	--model_name_or_path output/e-snli_t5_large_highlight \
	--beams 5 \
	--task wt5_DI_highlight
```

### Filtering Rationales

Following the collection step, train a classifier as proxy on e-SNLI dataset: `sh run_exnli.sh`

The filter final rationales collected from all sources: `run_esnli_instance_predict.sh`

### Final Rationale Generation Models

* train final rationale generation (on the filtered rationales collected from different sources)

Download and put the `final_rationale` folder in the `data/` folder, then run:

```
python -m source.generative.encoder_decoder \
	--train_file data/final_rationale/train.csv \
	--eval_data_file data/final_rationale/dev.csv \
	--out_dir output/final_rationale_bart-large \
	--model_type bart-large \
	--model_name_or_path bart-large \
	--device 0 \
	--do_train \
	--save_steps 1000 \
	--save_total_limit 1 \
	--num_train_epochs 2 \
	--logging_steps 2000 \
	--gradient_accumulation_steps 8 \
	--train_batch_size 64 \
	--task rationale
```

NOTE: `--model_type` can be among ["gpt2-xl", "bart-large"], and `--task` can be among ["rationale", "multi", "update_rationale", "update_type_rationale"].

Then generate rationales using:

```
python -m source.generative.generate_texts \
	--in_file data/final_rationale/test.csv \
	--out_file output/rationale_bart-large/test_rationale_bart-large.jsonl \
	--model_name_or_path output/rationale_bart-large \
	--beams 5 \
	--task rationale \
	--device 0
```

#### How to cite?

```
@inproceedings{brahman2021rationalize,
  title={Learning to Rationalize for Nonmonotonic Reasoning with Distant Supervision},
  author={Faeze Brahman, Vered Shwartz, Rachel Rudinger and Yejin Choi},
  booktitle={AAAI},
  year={2021}
}
```
