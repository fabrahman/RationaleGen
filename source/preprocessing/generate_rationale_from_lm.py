import json
import tqdm
import torch
import spacy
import string
import logging
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


COMMON_VERBS = ["be", "get", "have", "take", "go", "come", "give", "need", "can", "could", "will", "would", "may", "might", "do", "shall"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--dataset_type", default="winogrande", type=str, required=False,
                        help="base dataset format (winogrande, socialiqa, commonsenseqa, mctaco or copa)")
    parser.add_argument("--out_file", default=None, type=str, required=True, help="Output jsonl file")
    parser.add_argument("--lm", default="gpt2", type=str, required=False, help="Which language model to use")
    parser.add_argument("--device", default=-1, type=int, required=False, help="GPU device")
    parser.add_argument("--max_rationale_length", default=25, type=int, required=False,
                        help="max rationale length in words")
    parser.add_argument("--p_sampling_questions", default=0.0, type=float, required=False,
                        help="p for top_p for questions")
    parser.add_argument("--k_sampling_questions", default=0, type=int, required=False, help="k for top_k for questions")
    parser.add_argument("--rationale_redundancy", default=25, type=int, required=False,
                        help="how many rationale to generate from each prefix")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
                        help="BERT model to rank the clarifications")
    parser.add_argument('--max_clarifications', default=3, type=int, help="how many clarifications to keep")

    args = parser.parse_args()
    logger.info(args)

    nlp = spacy.load('en_core_web_sm')

    device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device("cpu")
    generator = LMTextGenerator(args.lm, device=device)

    # BERT is used to replace the placeholder in WinoGrande with a pronoun.
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    bert_model = BertForMaskedLM.from_pretrained(args.bert_model).to(device)

    # # Prefixes of clarification questions and their corresponding answers.
    # prefixes = json.load(open(args.prefixes_file, 'r'))
    #
    # num_lines = sum(1 for _ in open(args.dataset))

    df = pd.read_csv(args.dataset, sep=",").replace({np.nan: None}).to_dict('index')
    num_lines = len(df)

    with open(args.out_file, "w") as f_out:
        logger.info(f"Reading instances from lines in file at: {args.dataset}")
        for _, line in tqdm.tqdm(df.items(), total=num_lines):

            # Get the context and the choices
            if args.dataset_type == 'definf-snli':
                premise = line['Input_premise']
                hypothesis = line['Input_hypothesis'].strip(".")

            else:
                assert (False, "Dataset should be one of snli, ATOMIC, social-norm")

            roles = [('Intensifier', "more"), ('Attenuator', "less")]

            for role, mode in roles:
                if line[f'Answer_{role}_modifier'] != None:
                    content_words = get_content_words(line[f'Answer_{role}_modifier'], nlp)
                    line[f"{role}_lm_rationale"] = generate_clarifications(
                        generator, premise, hypothesis, line[f'Answer_{role}_modifier'], content_words, mode,
                        args.max_rationale_length,
                        p_sampling_questions=args.p_sampling_questions,
                        k_sampling_questions=args.k_sampling_questions,
                        rationale_redundancy=args.rationale_redundancy,
                    ) # if line[f'Answer_{role}_impossible'] !="on" else []
                # elif line[f'Answer_{role}_impossible'] == "on":
                elif line[f'Answer_{role}_modifier'] == None:
                    line[f"{role}_lm_rationale"] = []

            # if "_" in context:
            #     substitute = get_best_pronoun(bert_model, bert_tokenizer, device, context)
            #     context = context.replace("_", substitute)

            # curr_clarifications = generate_clarifications(
            #     generator, choices, args.max_clarification_question_length,
            #     args.max_answer_length, context, prefixes, args.dataset_type,
            #     question=fields["question"] if args.dataset_type in {"socialiqa", "mctaco"} else "",
            #     p_sampling_questions=args.p_sampling_questions,
            #     k_sampling_questions=args.k_sampling_questions,
            #     p_sampling_answers=args.p_sampling_answers,
            #     k_sampling_answers=args.k_sampling_answers,
            #     question_redundancy=args.question_redundancy,
            #
            # fields['clarifications'] = curr_clarifications + [('None', 'None')]

            # del line["Attenuator_content_words"]
            # del line["Intensifier_content_words"]

            f_out.write(json.dumps(line) + '\n')
            f_out.flush()


def get_content_words(text, nlp):
    """
    Return all the adjectives, nouns and verbs in the text.
    """
    doc = nlp(text)
    content_words = []
    # for t in doc:
    #     if t.pos_ in {"VERB"} and t.lemma_ not in COMMON_VERBS and not t.is_stop:
    #         content_words.append(t.lemma_)
    #     elif t.pos_ in {"NOUN", "ADJ"} and not t.is_stop:
    #         content_words.append(t.text)
    content_words = [t.text for t in doc if (t.pos_ in {"NOUN"} and not t.is_stop)]
#    content_words = [t.lemma_ for t,n in zip(content_words,doc) if n.pos_ in {"VERB"}]
    return list(set(map(str.lower, content_words)))


def generate_clarifications(generator: LMTextGenerator,
                            premise: str,
                            hypothesis: str,
                            update: str,
                            content_words: List[str],
                            mode: str,
                            max_rationale_length: int,
                            p_sampling_questions: Optional[float] = 0.0,
                            k_sampling_questions: Optional[int] = 0,
                            rationale_redundancy: int = 25):
    """
    Generate multiple clarification questions and answers them.

    generator: the language model.
    choices: the choices for multiple choice question answers.
    max_rationale_length: the maximum number of tokens in a clarification question.
    p_sampling_questions: p for Nucleus sampling for the question.
    k_sampling_questions: k for top k sampling for the question.
    rationale_redundancy: how many questions to generate.

    Returns:
        A list of (question, answer)
    """
    # question_prefixes = [k for k in prefixes.keys() if not k.endswith("?")]

    # Generate the rationales
    # print(content_words)
    content_words += [""]
    input_context = f"{premise} If {update.lower().strip('.')}, then it's {mode} likely that {hypothesis.lower()} because"
    generated_rationales = generator.generate([" ".join((input_context, cont_w))
        for cont_w in content_words],
        length=max_rationale_length, stop_token='.',
        p=p_sampling_questions, k=k_sampling_questions, num_samples=rationale_redundancy)

    # print(generated_rationales)

    # generated_rationales = list(generated_rationales.values())
    # print(generated_rationales)

    # Filter out short clarifications
    # words = lambda s: set(s.translate(str.maketrans('', '', string.punctuation)).split())

    generated_rationales = [' '.join((input_context, cont_w, clar_q))
                               for cont_w in content_words
                               for i, clar_qs in generated_rationales.items()
                               for clar_q in clar_qs
                               ] # if len(words(clar_q).intersection(words(input_context))) >= 1

    generated_rationales = list(set(generated_rationales))


    return generated_rationales


def get_best_pronoun(bert_model, bert_tokenizer, device, context):
    """
    Replaces the placeholder with the most likely pronoun
    """
    subs = ["he", "she", "it", "they", "her", "him", "them", "thing", "one", "someone", "ones", "things"]
    subs = sorted(
        zip(subs, get_substitute_probabilities(bert_model, bert_tokenizer, context, subs, device)),
        key=lambda item: item[-1], reverse=True)
    substitute = subs[0][0]
    return substitute


def generate_answers(
        context: str,
        clarification_questions: List,
        generator: LMTextGenerator,
        max_answer_length: int,
        answer_redundancy: int = 3,
        p_sampling_answers: Optional[float] = 0.0,
        k_sampling_answers: Optional[int] = 0):
    """
    Generate answers for the clarification questions.

    context: the context.
    clarification_questions: list of generated clarification questions.
    generator: the language model.
    max_answer_length: he maximum number of tokens in an answer.
    answer_redundancy: how many answers to generate.
    p_sampling_answers: p for Nucleus sampling for the answer.
    k_sampling_answers: k for top k sampling for the answer.

    Returns:
        A list of (question, answer)
    """
    words = lambda s: set(s.translate(str.maketrans('', '', string.punctuation)).split())
    generation_prefixes = [" ".join((context, answer_prefix)) for _, answer_prefix in clarification_questions]
    answers = generator.generate(generation_prefixes, length=max_answer_length,
                                 stop_token='.', p=p_sampling_answers, k=k_sampling_answers,
                                 num_samples=max(1, answer_redundancy))

    if len(answers) == 0:
        return []

    _, answers = zip(*sorted(answers.items(), key=lambda x: x[0]))

    # Filter out short answers.
    capitalize = lambda s: s[0].upper() + s[1:]

    clar_questions_and_answers = [(clar_q, ' '.join([s for s in [capitalize(ans_prefix) if len(ans_prefix) > 0 else '',
                                                                 answer.replace("..", ".")] if s is not '']))
                                  for (clar_q, ans_prefix), curr_answers in zip(clarification_questions, answers)
                                  for answer in curr_answers
                                  if len(words(answer.lower()).intersection({"i", "i'm", "you", "your", "me"})) == 0]

    # Save one instance of every answer and add empty answer, i.e. not using any of the generated answers.
    clar_questions_and_answers = {ans: clar_q for clar_q, ans in clar_questions_and_answers}
    clar_questions_and_answers = list(set([(clar_q, ans) for ans, clar_q in clar_questions_and_answers.items()]))
    return clar_questions_and_answers


def get_answer_prefix(prefixes=None,
                      question_prefix: str = None,
                      question: str = None):
    """
    Returns the answer prefix for each question
    """
    question_prefix = question_prefix.replace("Question:", "").strip()
    answer_prefix = prefixes.get(question_prefix, "")
    answer_prefix = answer_prefix.replace("_", question.replace("?", ""))
    return answer_prefix


def get_substitute_probabilities(bert_model, bert_tokenizer, text, choices, device):
    """
    Find the best pronoun to replace the placeholder, using BERT.
    """
    choices_indices = [bert_tokenizer.convert_tokens_to_ids(
        bert_tokenizer.tokenize(choice))[0] for choice in choices]
    text = " ".join(("[CLS]", text.replace("_", "[MASK]"), "[SEP]"))

    # Tokenize input
    tokenized_text = bert_tokenizer.tokenize(text)
    masked_index = [i for i, token in enumerate(tokenized_text) if token == "[MASK]"][0]

    # Convert token to vocabulary indices
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).long().to(device)
    segments_tensors = torch.tensor([np.ones(len(indexed_tokens))]).long().to(device)

    bert_model.eval()

    with torch.no_grad():
        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

    predictions = F.softmax(predictions[0, masked_index], dim=-1)

    # Compute the probability of the choices
    probs = [predictions[choice] for choice in choices_indices]

    return probs


if __name__ == '__main__':
    main()
