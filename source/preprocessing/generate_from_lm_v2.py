import json
import tqdm
import torch
import spacy
import string
import logging
import itertools
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from typing import Optional, List, Dict
from transformers import BertTokenizer, BertForMaskedLM
from source.preprocessing.lm_text_generator import LMTextGenerator

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
# nlp = spacy.load('en_core_web_lg')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--dataset_type", default="winogrande", type=str, required=False,
                        help="base dataset format (winogrande, socialiqa, commonsenseqa, mctaco or copa)")
    parser.add_argument("--out_file", default=None, type=str, required=True, help="Output jsonl file")
    parser.add_argument("--lm", default="gpt2", type=str, required=False, help="Which language model to use")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")
    parser.add_argument("--max_single_rationale_length", default=10, type=int, required=False,
                        help="max rationale length in words")
    parser.add_argument("--max_pair_rationale_length", default=15, type=int, required=False,
                        help="max rationale length in words")
    parser.add_argument("--p_sampling", default=0.0, type=float, required=False,
                        help="p for top_p for questions")
    parser.add_argument("--k_sampling", default=0, type=int, required=False, help="k for top_k for questions")
    parser.add_argument("--rationale_redundancy", default=25, type=int, required=False,
                        help="how many rationale to generate from each prefix")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
                        help="BERT model to rank the clarifications")
    parser.add_argument('--max_clarifications', default=3, type=int, help="how many clarifications to keep")

    args = parser.parse_args()
    logger.info(args)

    # nlp = spacy.load('en_core_web_sm')
    nlp = spacy.load('en_core_web_lg')

    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    generator = LMTextGenerator(args.lm, device=device)

    prefixes = {
        "definition": ["What is the definition of \"{}\"?", "The definition of \"{}\" is"],
        "purpose": ["What is the purpose of \"{}\"?", "The purpose of \"{}\" is"],
        "relation": ["What is the relationship between \"{}\" and \"{}\"?", "The relationship between \"{}\" and \"{}\" is"],
        "difference": ["What is the difference between \"{}\" and \"{}\"?", "The difference between \"{}\" and \"{}\" is"]
    }
    fields_to_remove = ["update_extracted_rationale", "hypothesis_extracted_rationale", "update_spans", "hypothesis_spans",
                        "update_Vphrases", "update_Nphrases", "hypothesis_Vphrases", "hypothesis_Nphrases"]

    # prefixes = json.load(open(args.prefixes_file, 'r'))

    num_lines = sum(1 for _ in open(args.dataset))

    with open(args.dataset, "r") as f_in:
        with open(args.out_file, "w") as f_out:
            logger.info(f"Reading instances from lines in file at: {args.dataset}")
            for line in tqdm.tqdm(f_in, total=num_lines):
                fields = json.loads(line.strip())

                # Get the premise, update and hypothesis
                if args.dataset_type == 'definf-snli':
                    # premise, update, hypothesis = segment_input(fields["input"], nlp)
                    premise, update, hypothesis = fields["premise"] , fields["update"], fields["hypothesis"]
                else:
                    assert (False, "Dataset should be one of snli, ATOMIC, social-norm")

                update_queries = fields["upd_verbs"] + fields["upd_nouns"]
                hypothesis_queries = fields["hypo_verbs"] + fields["hypo_nouns"]
                pair_queries = list(itertools.product(update_queries, hypothesis_queries))
                pair_queries = sorted(
                    pair_queries,
                    key=lambda pair: similarity(nlp, *pair),
                    reverse=True
                )[:3]

                fields["update_single_rationale"] = generate_single_clarifications(
                    generator, premise, prefixes,
                    args.max_single_rationale_length,
                    query_v=fields["upd_verbs"],
                    query_n=fields["upd_nouns"],
                    aux_context=update,
                    p_sampling=args.p_sampling,
                    rationale_redundancy=args.rationale_redundancy,
                )

                fields["hypothesis_single_rationale"] = generate_single_clarifications(
                    generator, premise, prefixes,
                    args.max_single_rationale_length,
                    query_v=fields["hypo_verbs"],
                    query_n=fields["hypo_nouns"],
                    aux_context=hypothesis,
                    p_sampling=args.p_sampling,
                    rationale_redundancy=args.rationale_redundancy,
                )

                fields["pair_rationales"] = generate_pair_clarifications(
                    generator, premise, prefixes,
                    args.max_pair_rationale_length,
                    query_pair=pair_queries,
                    aux_context="",
                    p_sampling=args.p_sampling,
                    rationale_redundancy=args.rationale_redundancy,
                )

                for item in fields_to_remove:
                    del fields[item]

                #             print(
                #                 fields["update_single_rationale"] +
                #                 fields["hypothesis_single_rationale"]) # +
                #                 fields["pair_rationales"])
                f_out.write(json.dumps(fields) + '\n')
                # f_out.flush()


def generate_single_clarifications(generator: LMTextGenerator,
                                   premise: str,
                                   prefixes: Dict[str, List[str]],
                                   max_rationale_length: int,
                                   query_v: List[str],
                                   query_n: List[str],
                                   aux_context: str = "",
                                   p_sampling: Optional[float] = 0.0,
                                   k_sampling: Optional[int] = 0,
                                   rationale_redundancy: int = 5):
    """
    Generate multiple rationale given query lists.

    generator: the language model.
    premise:
    prefixes: the "definition" and "purpose" prefixes for single phrase queries
    max_rationale_length: the maximum number of tokens in a clarification question.
    query_v: list of verb phrases
    query_n: list of noun phrases
    aux_context: auxilary context from update or hypothesis
    p_sampling_questions: p for Nucleus sampling for the question.
    k_sampling_questions: k for top k sampling for the question.
    rationale_redundancy: how many questions to generate.

    Returns:
        A list of (prefix question, rationale)
    """
    # Generate the rationales
    if len(query_n) == 0 and len(query_v) == 0:
        return []

    context = " ".join((premise, aux_context))

    final_rationale = []
    for key, query in zip(["definition", "purpose"], (query_n, query_v)):
        if len(query) == 0:
            continue

        generated_rationales = generator.generate([
            " ".join((context, prefixes[key][1].format(cont_w)))
            for cont_w in query],
            length=max_rationale_length,
            p=p_sampling,
            num_samples=rationale_redundancy)

        generated_rationales = [
            (prefixes[key][0].format(query[i]), ' '.join((prefixes[key][1].format(query[i]), clar_q)))
            for i, clar_qs in generated_rationales.items()
            for clar_q in clar_qs]

        final_rationale += generated_rationales

    final_rationale = list(set(final_rationale))

    return final_rationale


def generate_pair_clarifications(generator: LMTextGenerator,
                                 premise: str,
                                 prefixes: Dict[str, List[str]],
                                 max_rationale_length: int,
                                 query_pair: List[tuple],
                                 aux_context: str = "",
                                 p_sampling: Optional[float] = 0.0,
                                 k_sampling: Optional[int] = 0,
                                 rationale_redundancy: int = 5):
    """
    Generate multiple clarification questions and answers them.

    generator: the language model.
    prefixes: the "relation" and "difference" prefixes for paired phrase queries
    max_rationale_length: the maximum number of tokens in a clarification question.

    p_sampling_questions: p for Nucleus sampling for the question.
    k_sampling_questions: k for top k sampling for the question.
    rationale_redundancy: how many questions to generate.

    Returns:
        A list of (prefix question, rationale)
    """
    if len(query_pair) == 0:
        return []

    context = premise

    final_rationale = []
    for key in ["relation", "difference"]:
        prompts = [" ".join((context,
                             prefixes[key][1].format(cont_w1, cont_w2),
                             f"that {cont_w1}"))
                   for cont_w1, cont_w2 in query_pair]

        generated_rationales = generator.generate(
            prompts,
            length=max_rationale_length,
            p=p_sampling,
            num_samples=rationale_redundancy)

        generated_rationales = [
            # Question
            (prefixes[key][0].format(query_pair[i][0], query_pair[i][1]),
             ' '.join(
                 (f"{prefixes[key][1].format(query_pair[i][0], query_pair[i][1])} that {query_pair[i][0]}", clar_q)))
            for i, clar_qs in generated_rationales.items()
            for clar_q in clar_qs
        ]

        final_rationale += generated_rationales

    final_rationale = list(set(final_rationale))

    return final_rationale

def similarity(nlp, x, y):
    doc_x, doc_y = [nlp(w) for w in [x, y]]
    return max(
        [w_x.similarity(w_y)
         if w_x and w_x.vector_norm and w_y and w_y.vector_norm
         else 0.0
         for w_x in doc_x
         for w_y in doc_y])


# def segment_input(input_seq, nlp):
#     input_seq = input_seq.strip("<s> ").strip(" </s>")
#     prem_upd, hypothesis = input_seq.split("</s></s> ")[0], input_seq.split("</s></s> ")[1]
#
#     doc  = nlp(prem_upd)
#     premise, update = [sentence.text for sentence in doc.sents]
#
#     return premise, update, hypothesis


if __name__ == '__main__':
    main()
