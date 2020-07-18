import os
import json
import tqdm
import spacy
import logging
import argparse
import itertools
import pandas as pd

from source.preprocessing.conceptnet_helper import build_conceptnet, load_conceptnet, shortest_paths, to_natural_language

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


COMMON_VERBS = ["be", "get", "have", "take", "go", "come", "give", "need", "can", "could", "will", "would", "may", "might", "do", "shall"]
DISALLOWED_RELS = ['relatedto', 'relatedto-1', "formof", "mannerof", "hascontext", "definedas", "derivedfrom", "dbpedia", "receivesaction", "etymologicallyrelatedto", "etymologicallyderivedfrom", "instanceof"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True, help="Jsonl file")
    parser.add_argument("--dataset_type", default="winogrande", type=str, required=False,
                        help="base dataset format (winogrande, socialiqa, commonsenseqa, mctaco, or copa)")
    parser.add_argument("--out_file", default=None, type=str, required=True, help="Output jsonl file")
    parser.add_argument("--answer_redundancy", default=3, type=int, required=False,
                        help="how many answers to generate from each question")
    parser.add_argument('--max_clarifications', default=20, type=int, help="how many clarifications to keep")
    parser.add_argument('--max_length', default=2, type=int, help="maximum path length in edges")
    parser.add_argument("--conceptnet_dir", default="~/resources/conceptnet", type=str, help="ConceptNet directory")

    args = parser.parse_args()
    logger.info(args)

    nlp = spacy.load('en_core_web_sm')

    num_lines = sum(1 for _ in open(args.dataset))

    logger.info("Initializing ConceptNet")
    conceptnet_dir = os.path.expanduser(args.conceptnet_dir)
    if not os.path.exists(os.path.join(conceptnet_dir, 'cooc.npz')):
        logger.info("ConceptNet not found, building it.")
        build_conceptnet(conceptnet_dir)

    conceptnet = load_conceptnet(conceptnet_dir)

    df = pd.read_csv(args.dataset, sep=",").to_dict('index')

    with open(args.out_file, "w") as f_out:
        logger.info(f"Reading instances from lines in file at: {args.dataset}")
        for i,( _, line) in enumerate(df.items()):
            if i % 10000 == 0:
                print("{} processed!".format(i))
            # Get pairs of concepts to query ConceptNet for their relationship
            if args.dataset_type == 'definf-snli':
                # context = ""
                # hypothesis = ""
                hypothesis = line['Input_hypothesis']
                hypothesis_content_words = get_content_words(hypothesis, nlp)
                line['Attenuator_conceptnet_supervision'] = ""
                line['Intensifier_conceptnet_supervision'] = ""
                if line['Answer_Attenuator_impossible'] != "on":
                    context = line['Answer_Attenuator_modifier'] # line['Input_premise'] + " " +

                    # Texts: any pair of content words from the context
                    context_content_words = get_content_words(context, nlp)
                    queries = list(set(list(set(itertools.product(context_content_words, hypothesis_content_words)))))
                    line['Attenuator_conceptnet_supervision'] = generate_clarifications(queries, conceptnet, answer_redundancy=args.answer_redundancy, max_length=args.max_length)

                if line['Answer_Intensifier_impossible'] != "on":
                    context = line['Answer_Intensifier_modifier'] # line['Input_premise'] + " " +

                    # Texts: any pair of content words from the context
                    context_content_words = get_content_words(context, nlp)
                    queries = list(set(list(itertools.product(context_content_words, hypothesis_content_words))))
                    line['Intensifier_conceptnet_supervision'] = generate_clarifications(queries, conceptnet, answer_redundancy=args.answer_redundancy, max_length=args.max_length)

            else:
                assert(False, "Dataset should be one of snli, social_norm")

        # curr_clarifications = generate_clarifications(
        #     queries, conceptnet, answer_redundancy=args.answer_redundancy, max_length=args.max_length)
        #
        # fields['clarifications'] = curr_clarifications + [("None", "None")]

            f_out.write(json.dumps(line) + '\n')
            f_out.flush()

# older
# def get_content_words(text, nlp):
#     """
#     Return all the adjectives, nouns and verbs in the text.
#     """
#     doc = nlp(text)
#     content_words = []
#     for t in doc:
#         if t.pos_ in {"VERB"}:
#             content_words.append(t.lemma_)
#         elif t.pos_ in {"NOUN", "ADJ", "ADV"}:
#             content_words.append(t.text)
# #    content_words = [t.text for t in doc if t.pos_ in {"VERB", "NOUN", "ADJ"}]
# #    content_words = [t.lemma_ for t,n in zip(content_words,doc) if n.pos_ in {"VERB"}]
#     return list(set(map(str.lower, content_words)))

def get_content_words(text, nlp):
    """
    Return all the adjectives, nouns and verbs in the text.
    """
    doc = nlp(text)
    content_words = []
    for t in doc:
        if t.pos_ in {"VERB"} and t.lemma_ not in COMMON_VERBS and not t.is_stop:
            content_words.append(t.lemma_)
        elif t.pos_ in {"NOUN", "ADJ", "ADV"} and not t.is_stop:
            content_words.append(t.text)
    # content_words = [t.text for t in doc if (t.pos_ in {"NOUN"} and not t.is_stop)]
#    content_words = [t.lemma_ for t,n in zip(content_words,doc) if n.pos_ in {"VERB"}]
    return list(set(map(str.lower, content_words)))


def generate_clarifications(word_pairs, conceptnet, answer_redundancy=3, max_length=3):
    """
    Find ConceptNet paths between each word pair.
    word_pairs: list of word pairs.
    conceptnet: an initialized `Resource` object.
    answer_redundancy: how many paths to keep.
    max_length: how many edges in a path.
    """
    results = {f'What is the relationship between "{w1}" and "{w2}"?':
                   shortest_paths(
                       conceptnet, w1, w2, max_length=max_length, exclude_relations=DISALLOWED_RELS)
               for w1, w2 in word_pairs}

    # Prune
    results = {question: answers for question, answers in results.items() if len(answers) > 0}
    results = [(answer, weight)
               for _, answers in results.items()
               for answer, weight in answers
               if len(answer) > 0]

    results = [(to_natural_language(answer))
               for answer, weight in sorted(results, key=lambda x: x[-1], reverse=True)[:answer_redundancy]]

    return results


if __name__ == '__main__':
    main()
