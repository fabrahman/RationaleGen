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
    df = pd.read_csv(in_file)

    # Determine the names of the premise and hypothesis columns
    columns = set(df.columns)

    # SC
    if "Input_rot" in columns:
        premise_col, hypo_cols = None, ["Input_rot"]
    # SNLI
    elif "Input_premise" in columns:
        assign_col, premise_col, hypo_cols, id_col = "AssignmentId", "Input_premise", ["Input_hypothesis"], "Input_pairID"
    # ATOMIC
    elif "Input_event" in columns:
        premise_col, hypo_cols = "Input_event", [f"Input_{rel}" for rel in get_atomic_relations()]
    elif "Sentence1" in columns:
        premise_col, hyp_cols, label_col, exp_col = "Sentence1", "Sentence2", "gold_label", "Explanation_1"
    else:
        raise ValueError("Wrong data format, missing premise and hypothesis columns")

    # e-snli dataset
    if "Sentence1" in columns:
        examples = [
            (
                "explain nli premise: {} hypothesis: {}".format(row[premise_col], row[hyp_cols]),
                "{} explanation: {} <eos>".format("intensifier" if row[label_col]=="entailment" else "attenuator", row[exp_col])
            )
            for ind, row in df.iterrows()
        ]

    # defeasible datasets
    else:
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
            # classification from scratch on definf data
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

