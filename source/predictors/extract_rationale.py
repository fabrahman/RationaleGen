import os
import json
import numpy as np
import logging
import argparse
import tqdm
import math
import spacy
import textacy
import re

logger = logging.getLogger(__name__)


def remove_det(phrase, nlp):
    pattern = [
        {'POS': 'DET', 'OP': '+'},
        {'POS': 'ADJ', 'OP': '*'},
        {'POS': 'NOUN', "OP": '+'}
    ]
    doc = textacy.make_spacy_doc(phrase, lang='en_core_web_sm')
    # if phrase.startswith("an ") or phrase.startswith("a ") or phrase.startswith("the "):
    matches = [phs.text for phs in textacy.extract.matches(doc, pattern)]
    if len(matches) != 0 and phrase.find(matches[0]) == 0: # matched phrase should be in the beginning
        return phrase.split(" ",1)[1]
    return phrase

def get_longest_vphrase(matches_):
    seen_se = set()
    filtered_matches = []
    for match in sorted(matches_, key=len, reverse=True):
        s, e = match.start, match.end
        if any(s >= ms and e <= me for ms, me in seen_se):
            continue
        else:
            seen_se.add((s, e))
            filtered_matches.append(match)
    final_list = sorted(filtered_matches, key=lambda m: m.start)
    final_list = [phrase.text for phrase in final_list]
    return final_list

def get_verb_phrases(text, nlp):
    verb_pattern_1 = [{"POS": "VERB", "OP": "*"}, {"POS": "ADV", "OP": "*"}, {"POS": "VERB", "OP": "+"},
                    {"POS": "ADP", "OP": "*"}, {"POS": "DET", "OP": "*"}, {"POS": "NOUN", "OP": "+"}]
    verb_pattern_2 = [{"POS": "VERB", "OP": "*"}, {"POS": "ADV", "OP": "*"}, {"POS": "VERB", "OP": "+"},
                    {"POS": "ADP", "OP": "*"}]
    # verb_pattern_3 = [{"POS": "AUX", "OP":"+"},  {"POS": "ADV", "OP": "*"}, {"POS":"ADJ", "OP":"+"}] # for phrases like: she `is happy`. Later I decided to extract ADJ in noun phrases instead.

    doc = textacy.make_spacy_doc(text, lang='en_core_web_sm')
    verb_phrases_1 = get_longest_vphrase(list(textacy.extract.matches(doc, verb_pattern_1)))
    verb_phrases_2 = get_longest_vphrase(list(textacy.extract.matches(doc, verb_pattern_2)))
    # verb_phrases_3 = list(textacy.extract.matches(doc, verb_pattern_3))

    all_verb_phrases = list(set(verb_phrases_1 + verb_phrases_2 )) # + verb_phrases_3
    # print(all_verb_phrases)
    return all_verb_phrases #get_longest_vphrase(verb_phrases)

def get_noun_phrases(text, nlp):
    doc = textacy.make_spacy_doc(text, lang='en_core_web_sm')
    # print([chunk.text for chunk in doc.noun_chunks])
    noun_phrase = [chunk.text for chunk in doc.noun_chunks] # noun phrase
    single_noun = [word.text for word in doc if (word.pos_ in ["NOUN", "ADJ", "ADV"] and not word.is_stop)] # single NOUN, ADJ, ADV

    all_noun_phrases = list(set(noun_phrase + single_noun))
    return all_noun_phrases

def match_verb_phrases(extracted, phrases, nlp):
    """
    match extracted phrases (from classifier attentions) that are matched with full/single noun/verb phrases obtained using spacy parser
    Args:
        extracted (list): list of spans obtained from topk attended spans by classifier
        phrases (list): noun/verb phrases obtained from spacy/textacy parser.
        nlp
    Return:
        result (list): list of phrases to query LM for definitions/relationships.
    """

    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    flatten_tokens = [tok for sublist in [span.split() for span in extracted] for tok in sublist if tok not in spacy_stopwords]
    # remove punctuations
    flatten_tokens = [re.sub(r'[^\w\s]', '', tok) for tok in flatten_tokens]
    result = []
    for tok in flatten_tokens:
        for phrase in phrases:
            if phrase.startswith(tok):
                phrase = remove_det(phrase, nlp)
                result.append(phrase.lower())

    return list(set(result))

def match_noun_phrases(extracted, phrases, nlp):
    """
    match extracted phrases (from classifier attentions) that are matched with full/single noun/verb phrases obtained using spacy parser
    Args:
        extracted (list): list of spans obtained from topk attended spans by classifier
        phrases (list): noun/verb phrases obtained from spacy/textacy parser.
        nlp
    Return:
        result (list): list of phrases to query LM for definitions/relationships.
    """

    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    flatten_tokens = [tok for sublist in [span.split() for span in extracted] for tok in sublist if tok not in spacy_stopwords]
    # remove punctuations
    flatten_tokens = [re.sub(r'[^\w\s]', '', tok) for tok in flatten_tokens]
    result = []
    for tok in flatten_tokens:
        for phrase in phrases:
            if tok in phrase:
                phrase = remove_det(phrase, nlp)
                result.append(phrase.lower())

    return list(set(result))

def get_spans(index_value):
    index_span = []
    value_span = []
    for i, (ind, val) in enumerate(index_value):
        if i == 0:
            index_span.append([ind])
            value_span.append([val])
            continue
        if ind == index_value[i-1][0] + 1:
            index_span[-1].append(ind)
            value_span[-1].append(val)
            continue
        index_span.append([ind])
        value_span.append([val])

    return index_span, value_span

def sort_value_and_indices(attentions, max_len):
    """
    take topk attentions with corresponding token indices. Sort indices to extract rationale from original sentence.
    Args:
        attentions: attention vectors
        max_len: maximum length of extracted indices
    return:
        sorted tuple (based on index) of (index_of_token, value_of_attention)
    """
    top_ind, top_vals = np.argsort(attentions)[-max_len:].tolist(), np.sort(attentions)[-max_len:]

    sorted_index_values = [(i, v) for i, v in sorted(zip(top_ind, top_vals), key=lambda pair: pair[0])]

    return sorted_index_values

def extract_rationale(args):

    in_file = os.path.join(args.data_dir, args.split+"_saliency.jsonl")
    out_file = os.path.join(args.output_dir, "{}_extracted_rationale{}.jsonl".format(args.split, "_sep" if args.separately else ""))
    num_lines = sum(1 for _ in open(in_file))

    nlp = spacy.load('en_core_web_sm')

    with open(in_file, "r") as f_in:
        with open(out_file, "w") as f_out:
            logger.info(f"Reading instances from lines in file at: {in_file}")
            for line in tqdm.tqdm(f_in, total=num_lines):
                fields = json.loads(line)

                attentions = np.array(fields["saliency"])
                input_tokens = fields["input"].split()
                assert (attentions.shape[0] == len(input_tokens))


                max_length = math.ceil(len(input_tokens) * args.topk)


                if not args.separately:
                    # sort attentions and then indices
                    sorted_ind_vals = sort_value_and_indices(attentions, max_length)
                    # print(sorted_ind_vals)
                    index_span, value_span = get_spans(sorted_ind_vals)
                    # fields["extracted_rationale"] =  [x for i, x in enumerate(input_tokens) if i in top_ind] #.replace("</s>", "").replace("<s>","")
                    fields["extracted_rationale"] = [" ".join(input_tokens[ind] for ind in span) for span in index_span]
                    # fields["spans"] = [{'span': (i, i + 1), 'value': float(round(v, 5))}
                    fields["spans"] = [{'span' : ind_span, 'value' : [float(round(v,5)) for v in val_span]} for ind_span, val_span in zip(index_span, value_span)]

                elif args.separately:
                    texta = fields["input"].split("</s></s> ")[0] # premise + update
                    texta_tokens = texta.split()
                    a_attentions = attentions[:len(texta_tokens)]
                    texta_sorted_ind_vals = sort_value_and_indices(a_attentions, max_length//2)
                    index_span_a, value_span_a = get_spans(texta_sorted_ind_vals)


                    textb = fields["input"].split("</s></s> ")[1] # hypothesis
                    textb_tokens = textb.split()
                    b_attentions = attentions[len(texta_tokens):]
                    assert len(b_attentions) == len(textb_tokens), "Error with different lengths for hypothesis and corresponding attention vector"
                    textb_sorted_ind_vals = sort_value_and_indices(b_attentions, max_length//2)
                    index_span_b, value_span_b = get_spans(textb_sorted_ind_vals)

                    fields["update_extracted_rationale"] = [" ".join(texta_tokens[ind] for ind in span) for span in index_span_a]
                    fields["hypothesis_extracted_rationale"] = [" ".join(textb_tokens[ind] for ind in span) for span in index_span_b]

                    fields["update_spans"] = [{'span' : ind_span, 'value' : [float(round(v,5)) for v in val_span]} for ind_span, val_span in zip(index_span_a, value_span_a)]
                    fields["hypothesis_spans"] = [{'span' :  [len(texta_tokens) + i for i in ind_span], 'value' : [float(round(v,5)) for v in val_span]} for ind_span, val_span in zip(index_span_b, value_span_b)] # [len(texta_tokens) + int(i) for i in ind_span]

                    # get noun and verb phrases from updates and hypothesis
                    if args.nv_phrases:
                        # doc_u = nlp(texta.strip(" <s> "))
                        # upd = [sentence.text for sentence in doc_u.sents][-1]
                        upd = fields["update"]
                        fields["update_Vphrases"] = get_verb_phrases(upd, nlp)
                        fields["update_Nphrases"] = get_noun_phrases(upd, nlp)

                        fields["hypothesis_Vphrases"] = get_verb_phrases(textb, nlp)
                        fields["hypothesis_Nphrases"] = get_noun_phrases(textb, nlp)

                        fields["upd_verbs"] = match_verb_phrases(fields["update_extracted_rationale"], fields["update_Vphrases"], nlp)
                        fields["upd_nouns"] = match_noun_phrases(fields["update_extracted_rationale"],
                                                                fields["update_Nphrases"], nlp)

                        fields["hypo_verbs"] = match_verb_phrases(fields["hypothesis_extracted_rationale"], fields["hypothesis_Vphrases"], nlp)
                        fields["hypo_nouns"] = match_noun_phrases(fields["hypothesis_extracted_rationale"],
                                                                fields["hypothesis_Nphrases"], nlp)


                del fields["saliency"]
                del fields["input"]

                f_out.write(json.dumps(fields) + '\n')
                f_out.flush()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .jsonl files (or other data files) for the task.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output dir to save results (jsonl files). ",
    )
    parser.add_argument("--split", default="", type=str, help="Which split of data train/dev/test")
    parser.add_argument("--topk", default=0.2, type=float, help="extraction max length ratio?")
    parser.add_argument("--separately", action="store_true", help="If extract rationales from premise-update and hyp separately.")
    parser.add_argument("--nv_phrases", action="store_true",
                        help="Whether to extract noun/verb phrases from input.")



    args = parser.parse_args()


    args.output_dir = os.path.join(args.output_dir, "{}{}".format(str(args.topk), "-sep" if args.separately else ""))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    extract_rationale(args)




if __name__ == "__main__":
    main()