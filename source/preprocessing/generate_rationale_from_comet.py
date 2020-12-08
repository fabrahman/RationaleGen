import re
import tqdm
import json
import spacy
import pandas as pd
import numpy as np
import textacy
import logging
import argparse

from comet2.comet_model import PretrainedCometModel


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


CATEGORY_TO_QUESTION = {"xIntent": "What was the intention of PersonX?",
                        "xNeed": "Before that, what did PersonX need?",
                        "oEffect": "What happens to others as a result?",
                        "oReact": "What do others feel as a result?",
                        "oWant": "What do others want as a result?",
                        "xEffect": "What happens to PersonX as a result?",
                        "xReact": "What does PersonX feel as a result?",
                        "xWant": "What does PersonX want as a result?",
                        "xAttr": "How is PersonX seen?"}

CATEGORY_TO_PREFIX = {"xIntent": "Because PersonX wanted",
                      "xNeed": "Before, PersonX needed",
                      "oEffect": "Others then",
                      "oReact": "As a result, others feel",
                      "oWant": "As a result, others want",
                      "xEffect": "PersonX then",
                      "xReact": "As a result, PersonX feels",
                      "xWant": "As a result, PersonX wants",
                      "xAttr": "PersonX is seen as"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--out_file", default=None, type=str, required=True, help="Output jsonl file")
    parser.add_argument("--device", type=str, required=False, default="cpu", help="cpu or GPU device")
    parser.add_argument("--model_file", type=str, required=False, help="The COMET pre-trained model", default=None)
    args = parser.parse_args()

    logger.info(f"Loading COMET model")

    # Load COMET either from args.model_file or from its default location.
    if args.model_file is not None:
        comet_model = PretrainedCometModel(model_name_or_path=args.model_file, device=args.device)
    else:
        comet_model = PretrainedCometModel(device=args.device)

    nlp = spacy.load('en_core_web_sm')

    # definf data is in csv format
    df = pd.read_csv(args.dataset, sep=",").replace({np.nan:None}).to_dict('index')
    df = df[(df["Answer_Intensifier_impossible"] != "on" | df["Answer_Attenuator_impossible"] != "on")]
    num_lines = len(df)
    roles = ["Hypothesis", 'Intensifier', 'Attenuator']
    with open(args.out_file, "w") as f_out:
        logger.info(f"Reading instances from lines in file at: {args.dataset}")
        for _, ex in tqdm.tqdm(df.items(), total=num_lines):
            for role in roles:
                ex[f"{role}_comet_supervision"] = get_rationales_definf_snli(ex, role, nlp, comet_model)
            f_out.write(json.dumps(ex) + "\n")


def get_rationales_definf_snli(ex, role, nlp, comet_model):
    """
    Generate clarifications for the defeasible-inference-snli dataset
    :param ex: a dictionary with the definf-snli instance
    :param role: strengthener or weakener or hypothesis?
    :param nlp: Spacy NLP
    :param comet_model: the COMET model
    :return: a list of (question, answer) tuples
    """

    CATEGORY_TO_TEMP = {"xIntent": "PersonX wanted {}",
                          "xNeed": "Before, PersonX needed {}",
                          "oEffect": "Others then {}",
                          "oReact": "As a result, others feel {}",
                          "oWant": "As a result, others want {}",
                          "xEffect": "PersonX then {}",
                          "xReact": "As a result, PersonX feels {}",
                          "xWant": "As a result, PersonX wants {}",
                          "xAttr": "PersonX is seen as {}"}

    premise = ex['Input_premise']

    if role == "Hypothesis":
        context = ex['Input_hypothesis']
        relevant_categories = ['xIntent', 'xNeed','xAttr']
        category_to_temp = {k: CATEGORY_TO_TEMP[k] for k in relevant_categories}

    # role in []
    else:
        update = ex[f'Answer_{role}_modifier']
        context = premise + " " + update
        relevant_categories = ['xEffect', 'xWant', 'xReact', 'xAttr', 'oEffect', 'oWant', 'oReact']
        category_to_temp = {k: CATEGORY_TO_TEMP[k] for k in relevant_categories}


    personx, is_named_entity = get_personx(nlp, context)
    personx = personx if (is_named_entity or personx == "I") else personx.lower()

    if len(personx) == 0:
        return []

    outputs = {category: comet_model.predict(context, category, num_beams=5) for category in relevant_categories}

    curr_events = []
    for category, prefix in category_to_temp.items():
        for out_event in outputs[category]:
            if out_event != "none" and out_event != "":
                if not out_event.lower().startswith("person") and not out_event.lower().startswith("other"):
                    out_event = prefix.format(out_event + ".")

                out_event = re.sub("personx", personx, out_event, flags=re.I)
                out_event = re.sub("person x", personx, out_event, flags=re.I)
                out_event = re.sub("persony", "others", out_event, flags=re.I)
                out_event = re.sub("person y", "others", out_event, flags=re.I)
                out_event = out_event.replace("then to", "then")

                # question = CATEGORY_TO_QUESTION[category].replace("PersonX", personx)
                curr_events.append(out_event)

    return curr_events

def get_personx(nlp, input_event, use_chunk=True):
    """
    Returns the subject of input_event
    """
    doc = nlp(input_event)
    svos = [svo for svo in textacy.extract.subject_verb_object_triples(doc)]

    if len(svos) == 0:
        if use_chunk:
            logger.warning(f'No subject was found for the following sentence: "{input_event}". Using noun chunks.')
            noun_chunks = [chunk for chunk in doc.noun_chunks]

            if len(noun_chunks) > 0:
                personx = noun_chunks[0].text
                is_named_entity = noun_chunks[0].root.pos_ == "PROP"
            else:
                logger.warning("Didn't find noun chunks either, skipping this sentence.")
                return "", False
        else:
            logger.warning(f'No subject was found for the following sentence: "{input_event}". Skipping this sentence')
            return "", False
    else:
        subj_head = svos[0][0]
        is_named_entity = subj_head.root.pos_ == "PROP"
        personx = " ".join([t.text for t in list(subj_head.lefts) + [subj_head] + list(subj_head.rights)])

    return personx, is_named_entity


if __name__ == "__main__":
    main()
