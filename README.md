## Unsupervised Rationale Generation for nonmonothonic Reasoning

### Generating Rationales

Follow the instruction to generate rationales from different resources:

1.  Pre-trained Language Model:

For faster generation, the train dataset is splitted and located at `data/defeasible-snli`

```bash
bash generate_lm_rationales.sh definf-snli data/defeasible-snli
```

2. COMET:

We used beam5 for decoding.

```
python source/preprocessing/generate_rationale_from_comet.py --dataset ./data/defeasible-snli/train.csv --dataset_type definf-snli --out_file ./data/defeasible-snli/comet_supervision/train_rationalized_comet.jsonl --device 2
```

3. ConceptNet:

For ConceptNet, we filtered out stopwords from hypothesis and updates, we took lemma for verbs, we also disallowed some common verbs. Additionally, we exclude some relations from conceptnet.

```
python generate_distant_supervision_from_conceptnet.py --dataset ./data/defeasible-snli/train.csv --dataset_type definf-snli --out_file ./data/defeasible-snli/conceptnet_suprvision/train_rationalized_conceptnet.jsonl --answer_redundancy 3 --max_length 2 --conceptnet_dir ~/resources/conceptnet
```

4. Pre-trained e-snli:

Here, we first fine-tune T5 in a WT5 format on e-snli dataset, in which we only used the instances that are labeled as 'contradiction` and `entailement`. The dataset is at `data/e-snli/` folder.

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
        --eval_batch_size 16
```

We then generate rationales for the train set of definf dataset using the pretrained rationale generation model on e-snli:
```
python -m source.generative.generate_texts \
	--in_file data/defeasible-snli/train.csv \
	--out_file output/e-snli_t5_large/train_definf_rationalized_ft_esnli.jsonl \
	--model_name_or_path output/e-snli-t5-large \
	-- beams 5 \
	--WT5 \
	--task rationale \
	--device 0
```

