import os
import json
import pandas as pd

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
    Currently supported: "rationale", "hypothesis", "update_type", "update", "update_rationale",
    "update_type_rationale", "multi" (all of them), "update_type_no_rationale" (original DI discriminative), 
    "update_no_rationale" (original DI generative), 
    "multi_no_rationale" (two previous ones)
    Returns a list of tuples (input, output)
    """
    df = pd.read_csv(in_file)

    # Always predict rationale
    values_to_predict = [value_to_predict]

    if value_to_predict == "multi":
        values_to_predict = ["rationale", "update_type", "update"]
    elif value_to_predict == "multi_no_rationale":
        values_to_predict = ["update_type_no_rationale", "update_no_rationale"]

    examples = [
        (
            f"[premise] {row['premise']} " +
            (f"[hypothesis] {row['hypothesis']} " if val != "hypothesis" else "") +
            (f"[update_type] <{row['update_type']}> " if "update_type" not in val else "") +
            (f"[update] {row['update']} " if val not in {'update', 'update_rationale', 'update_no_rationale'} else "") +
            (f"[rationale] {row['rationale']} " if "rationale" not in val else "") +
            f"[{val.replace('_no_rationale', '')}]",
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
    value_to_predict = value_to_predict.replace("_no_rationale", "")
    
    if value_to_predict in {"rationale", "hypothesis", "update"}:
        return row[value_to_predict]
    elif value_to_predict == "update_type":
        return f"<{row[value_to_predict]}>"
    elif value_to_predict == "update_rationale":
        return f"[update] {row['update']} [rationale] {row['rationale']}"
    elif value_to_predict == "update_type_rationale":
        return f"[update_type] <{row['update_type']}> [rationale] {row['rationale']}"
    
    return None


def load_data_wt5(in_file, task="wt5_esnli"):
    """
    Loads the dataset file:
    1) train on e-SNLI: Sentence1, Sentence2, gold_label, Explanation_1, (Highlight_tokens_1, Highlight_tokens_2)
    2) generate on DI:
        1- wt5_DI: Input_premise, Input_hypothesis, Answer_[role]_modifier (for role in {Attenuator, Intensifier})
        2- wt5_DI_highlight: update_extracted_rationale, hypothesis_extracted_rationale
    Returns the data in the format for training the generative model, i.e.:

    in_file: CSV file (wt5_esnli, wt5_esnli_highlight, wt5_DI) / jsonl file (wt5_DI_highlight)
    Returns a list of tuples (input, output)
    """
    if os.path.splitext(in_file)[-1] == ".csv":
        df = pd.read_csv(in_file)
    elif os.path.splitext(in_file)[-1] == ".jsonl" and task == "wt5_DI_highlight":
        data = []
        with open(in_file) as f_in:
            for line in f_in:
                field = json.loads(line)
                field["Highlight_tokens_1"] = " # ".join([tok for span in field["update_extracted_rationale"] for tok in span.split()])
                field["Highlight_tokens_2"] = " # ".join([tok for span in field["hypothesis_extracted_rationale"] for tok in span.split()])
                data.append(field)
        df = pd.DataFrame.from_records(data)
    columns = set(df.columns)

    # e-SNLI
    if "Sentence1" in columns:
        premise_col, hyp_col, label_col, exp_col, highlight1_col, highlight2_col = "Sentence1", "Sentence2", "gold_label", "Explanation_1", "Highlight_tokens_1", "Highlight_tokens_2"
    # DI
    elif "Input_premise" in columns:
        premise_col, hypo_col = "Input_premise", "Input_hypothesis"

    # train on esnli human-written rationales
    if task == "wt5_esnli":
        examples = [
            (
                "explain nli premise: {} hypothesis: {}".format(row[premise_col], row[hyp_col]),
                "{} explanation: {} <eos>".format("intensifier" if row[label_col] == "entailment" else "attenuator",
                                                  row[exp_col])
            )
            for _, row in df.iterrows()
        ]

    # generate rationales for DI using pre-trained wt5 on e-esnli
    elif task == "wt5_DI":
        df = df[df["Answer_Intensifier_impossible"] != "on"]
        df = df[df["Answer_Attenuator_impossible"] != "on"]

        # make sure the order of roles is the same in all sources.
        roles = ['Attenuator', 'Intensifier']
        modifier_cols = {role: f"Answer_{role}_modifier" for role in roles}
        examples = [
            (
                f"explain nli premise: {row[premise_col]} {row[modifier_cols[role]].strip('.')}. hypothesis: {row[hypo_col].strip('.')}.",
                f"{role.lower()} explanation: <eos>"
            )
            for _, row in df.iterrows()
            for role in roles
        ]

    # train on e-snli highlights
    elif task == "wt5_esnli_highlight":
        examples = [
            (
                "[premise] {} [hypo] {}".format(row[highlight1_col], row[highlight2_col]),
                "explanation: {} <eos>".format(row[exp_col])
            )
            for _, row in df.iterrows()
        ]

    # generate rationales for DI using pre-trained wt5 on e-snli highlights
    elif task == "wt5_DI_highlight":
        examples = [
            (
                "[premise] {} [hypo] {}".format(row[highlight1_col], row[highlight2_col]),
                "explanation: <eos>"
            )
            for _, row in df.iterrows()
        ]

    print("Example: ", examples[:2])
    print("Example: ", examples[-2:])
    return examples

