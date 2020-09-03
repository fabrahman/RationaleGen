import json
import argparse
import numpy as np

from nltk import bleu
from rouge import Rouge
from collections import defaultdict
from nltk.translate.bleu_score import SmoothingFunction

smoothing = SmoothingFunction().method1
weights = [0.25] * 4
rouge = Rouge()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, help="The directory of the outputs")
    args = parser.parse_args()

    print("\t".join(["Setup", "LM", "BLEU", "ROUGE"]))

    for setup in ["rationale", "multi", "update_rationale", "update_type_rationale"]:
        for lm in ["bart-large", "gpt2-xl"]:

            # Compute BLEU and ROUGE from the text predictions
            data = [json.loads(line.strip()) for line in open(f"{args.out_dir}/{setup}_{lm}/test_{setup}_{lm}.jsonl")]
            gold = defaultdict(list)
            predictions = defaultdict(set)

            for ex in data:
                curr_gold = ex["gold"].lower().replace("<eos>", "").strip()
                curr_preds = [pred.lower().strip() for pred in ex["predictions"]]
                curr_preds = set([pred for pred in curr_preds if len(pred) > 0])

                if len(curr_gold) > 0 and len(curr_preds) > 0:
                    gold[ex["input"]].append(curr_gold)
                    predictions[ex["input"]] = predictions[ex["input"]].union(curr_preds)

            bleu_scores, rouge_scores = [], []

            for input, curr_gold in gold.items():
                curr_predictions = list(predictions[input])

                # The refs and gold must be in the same size
                length = min(len(curr_gold), len(curr_predictions))

                if length > 0:
                    hyps = curr_predictions[:length]
                    refs = curr_gold[:length]
                    rouge_scores.extend([score["rouge-l"]["f"] for score in rouge.get_scores(hyps, refs)])

                    hyps = [tuple(h.split()) for h in hyps]
                    refs = [tuple(r.split()) for r in refs]
                    bleu_scores.extend([bleu(
                        refs, pred, weights=weights, smoothing_function=smoothing) for pred in hyps])

            print("\t".join([setup, lm, f"{100.0 * np.mean(bleu_scores):.3f}", f"{100.0 * np.mean(rouge_scores):.3f}"]))


if __name__ == "__main__":
    main()