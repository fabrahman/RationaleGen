import pandas as pd
import os
import json

from transformers import AutoModelWithLMHead, AutoTokenizer


def init_model(model_name: str, device, do_lower_case: bool = False):
    """
    Initialize a pre-trained LM
    :param model_name: from MODEL_CLASSES
    :param device: CUDA / CPU device
    :param do_lower_case: whether the model is lower cased or not
    :return: the model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_data_generative(in_file, value_to_predict="rationale"):
    """
    Loads the dataset file for the generative model, including the following columns:
    Original premise and hypothesis: Input_premise, Input_hypothesis
    Additional premise: for role in {Attenuator, Intensifier}:
    1) Answer_[role]_modifier: str, the additional premise.
    2) Answer_[role]_impossible: bool, on if Answer_[role]_modifier is None (ignore) or Nan otherwise.
    3) Answer_[role]_reason: str, reason that Answer_[role]_modifier modifies the original entailment (ignore).
    4) Answer_[role]_option: (nan, 'stereotyped', 'other').
    5) rationale
    Returns the data in the format for training the generative model
    in_file: CSV rot-details file
    value_to_predict: which item should be the output.
    Currently supported: "rationale", "hypothesis", "update_type", "update", "update_rationale", "multi" (all of them).
    Returns a list of tuples (input, output)
    """
    df = pd.read_csv(in_file)

    # Always predict rationale
    values_to_predict = [value_to_predict]


    if value_to_predict == "multi":
        values_to_predict = ["rationale", "update_type", "update"] #["rationale", "hypothesis", "update_type", "update", "update_rationale"]

    examples = [
        (
            f"[premise] {row['premise']} " +
            (f"[hypothesis] {row['hypothesis']} " if val != "hypothesis" else "") +
            (f"[update_type] <{row['update_type']}> " if "update_type" not in val else "") +
            (f"[update] {row['update']} " if val not in {'update', 'update_rationale'} else "") +
            (f"[rationale] {row['rationale']} " if "rationale" not in val else "") +
            f"[{val}]",
            f"{get_target(row, val)} <eos>"
        )
        for _, row in df.iterrows()
        for val in values_to_predict
    ]

    return examples


def get_target(row, value_to_predict):
    """
    Get the value of the field that needs to be predicted
    """
    if value_to_predict in {"rationale", "hypothesis", "update"}:
        return row[value_to_predict]
    elif value_to_predict == "update_type":
        return f"<{row[value_to_predict]}>"
    elif value_to_predict == "update_rationale":
        return f"[update] {row['update']} [rationale] {row['rationale']}"

    return None


def load_data(in_file, wt5=False, task="rationale"):
    """
    Loads the dataset file:
    Original premise and hypothesis:
    1) SC: Input_rot
    2) SNLI: Input_premise, Input_hypothesis
    3) ATOMIC: Input_event, Input_[rel] for rel in ATOMIC relations
    Additional premise: for role in {Attenuator, Intensifier}:
    1) Answer_[role]_modifier: str, the additional premise.
    2) Answer_[role]_impossible: bool, on if Answer_[role]_modifier is None (ignore) or Nan otherwise.
    3) Answer_[role]_reason: str, reason that Answer_[role]_modifier modifies the original entailment (ignore).
    4) Answer_[role]_option: (nan, 'stereotyped', 'other').
    Returns the data in the format for training the generative model, i.e.:
    If predicting the RoT:
    Input: "[premise] <premise> [hypo] <hypo> [attenuator]" or "[premise] <premise> [hypo] <hypo> [intensifier]"
    Output: Answer_Attenuator_modifier or Answer_Intensifier_modifier
    in_file: CSV rot-details file
    Returns a list of tuples (input, output)
    """
    if os.path.splitext(in_file)[-1] == ".csv":
        df = pd.read_csv(in_file)
    elif os.path.splitext(in_file)[-1] == ".jsonl" and task == "tokens-2-rationale":
        data = []
        with open(in_file) as f_in:
            for line in f_in:
                field = json.loads(line)
                field["Highlight_tokens_1"] = " # ".join([tok for span in field["update_extracted_rationale"] for tok in span.split()])
                field["Highlight_tokens_2"] = " # ".join([tok for span in field["hypothesis_extracted_rationale"] for tok in span.split()])
                data.append(field)
        df = pd.DataFrame.from_records(data)

    columns = set(df.columns)
    assign_col, premise_col, hypo_cols, id_col = "AssignmentId", "Input_premise", ["Input_hypothesis"], "Input_pairID"

    # e-SNLI
    if "Sentence1" in columns:
        premise_col, hyp_cols, label_col, exp_col = "Sentence1", "Sentence2", "gold_label", "Explanation_1"
        if "Highlight_tokens_1" in columns:
            highlight1_col, highlight2_col = "Highlight_tokens_1", "Highlight_tokens_2"

    # SNLI with extracted update/hypo spans (obtained from classifier attentions)
    elif "update_extracted_rationale" in columns:
        assign_col, highlight1_col, highlight2_col = "annotation_id", "Highlight_tokens_1", "Highlight_tokens_2" #"update_extracted_rationale", "hypothesis_extracted_rationale"

    # SNLI with final filtered rationales
    elif "rationale" in columns:
        assign_col, premise_col, update_col, hypo_col, role_col, rationale_col = "annotation_id", "premise", "update", "hypothesis", "update_type", "rationale"

    else:
        raise ValueError("Wrong data format, missing premise and hypothesis columns")

    # e-snli dataset (only for training)
    if "Sentence1" in columns:
        if task == "tokens-2-rationale":
            examples = [
                (
                    "[premise] {} [hypo] {}".format(row[highlight1_col], row[highlight2_col]),
                    "explanation: {} <eos>".format(row[exp_col])
                )
                for _, row in df.iterrows()
            ]

        # TODO: later use elif task == "wt5-esnli" ----> must change github readme
        else:

            examples = [
                (
                    "explain nli premise: {} hypothesis: {}".format(row[premise_col], row[hyp_cols]),
                    "{} explanation: {} <eos>".format("intensifier" if row[label_col]=="entailment" else "attenuator", row[exp_col])
                )
                for _, row in df.iterrows()
            ]

    # defeasible dataset only on highlighted spans
    elif task == "tokens-2-rationale":
        # rationale generation from highlighted (mostly attended) spans of update and hypothesis
        examples = [
            (
                "[premise] {} [hypo] {}".format(row[highlight1_col], row[highlight2_col]),#"[premise] {} [hypo] {}".format(" # ".join(row[highlight1_col].split())," # ".join(row[highlight2_col].split())),
                "explanation: <eos>",
                row[assign_col]
            )
            for _, row in df.iterrows()
        ]

    # defeasible dataset with final filtered rationales
    elif task == 'train-rat' or task == 'generate-rat':
        # rationale generation given all other inputs
        examples = [
            (
                "[premise] {} [update] {} [hypo] {} [{}] [rationale]".format(row[premise_col], row[update_col], row[hypo_col], row[role_col]),
                "{} <eos>".format(row[rationale_col]),
                row[assign_col]
            )
            for _, row in df.iterrows()
        ]

    # defeasible datasets
    else:
        df = df[~df["Answer_Intensifier_modifier"].isna()]
        df = df[~df["Answer_Attenuator_modifier"].isna()]

        roles = ['Intensifier', 'Attenuator']
        examples = {
            role: df[~df[f"Answer_{role}_modifier"].isna()][[f"Answer_{role}_modifier"] + hypo_cols + [assign_col] + [id_col] +
                                                          ([premise_col] if premise_col is not None else [])]
            for role in roles}
        modifier_cols = {role: f"Answer_{role}_modifier" for role in roles}
        # modeling WT5 format for rationale generation (only inference using pre-trained WT5 on e-snli) or classification (both training and testing)
        if wt5:
            # rationale generation
            if task == "rationale":
                examples = [
                    (
                        f"explain nli premise: {row[premise_col]} {row[modifier_cols[role]].strip('.')}. hypothesis: {row[hypo_col].strip('.')}.",
                        f"{role.lower()} explanation: <eos>",
                        ":".join([row[assign_col], row[id_col]])+":S" if role == 'Intensifier' else ":".join([row[assign_col], row[id_col]])+":W"
                    )
                    for role in roles
                    for _, row in examples[role].iterrows()
                    for hypo_col in hypo_cols
                ]

            elif task == "clf":
                examples = [
                    (
                        f"premise: {row[premise_col]} {row[modifier_cols[role]].strip('.')}. hypothesis: {row[hypo_col].strip('.')}.",
                        f"{role.lower()}",
                    )
                    for role in roles
                    for _, row in examples[role].iterrows()
                    for hypo_col in hypo_cols
                ]

        # Original Vered's part for generating updates
        # No premise
        elif premise_col is None:
            examples = [
                (
                    f"[hypo] {row[hypo_col]} [{role.lower()}]",
                    f"{row[modifier_cols[role]]} <eos>",
                )
                for role in roles
                for _, row in examples[role].iterrows()
                for hypo_col in hypo_cols
            ]

        # With a premise
        else:
            examples = [
                (
                    f"[premise] {row[premise_col]} [hypo] {row[hypo_col]} [{role.lower()}]",
                    f"{row[modifier_cols[role]]} <eos>",
                )
                for role in roles
                for _, row in examples[role].iterrows()
                for hypo_col in hypo_cols
            ]

    print("Example: ", examples[:2])
    print("Example: ", examples[-2:])
    return examples



def get_atomic_relations():
    return []
