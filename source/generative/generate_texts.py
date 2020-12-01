"""
Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
"""
import re
import json
import tqdm
import torch
import logging
import argparse

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

from source.generative.common import init_model, load_data, load_data_generative


def main() -> None:
    """
    Generate intensifiers and attenuators
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument(
        "--in_file",
        default=None,
        type=str,
        required=True,
        help="The input CSV file",
    )
    parser.add_argument(
        "--out_file",
        default=None,
        type=str,
        required=True,
        help="out jsonl file with generations",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        help="LM checkpoint for initialization.",
    )

    # Optional
    parser.add_argument(
        "--max_length", default=40, type=int, required=False, help="Maximum text length"
    )
    parser.add_argument(
        "--k", default=0, type=int, required=False, help="k for top k sampling"
    )
    parser.add_argument(
        "--p", default=0, type=float, required=False, help="p for nucleus sampling"
    )
    parser.add_argument(
        "--beams", default=0, type=int, required=False, help="beams for beam search"
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        required=False,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="GPU number or 'cpu'."
    )
    parser.add_argument(
        "--WT5", action="store_true", help="Whether to run generation in WT5 mode."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="what is the task when wt5 is on, rationale generation or , clf training?"
    )
    args = parser.parse_args()
    logger.debug(args)

    if (
        (args.k == args.p == args.beams == 0)
        or (args.k != 0 and args.p != 0)
        or (args.beams != 0 and args.p != 0)
        or (args.beams != 0 and args.k != 0)
    ):
        raise ValueError(
            "Exactly one of p, k, and beams should be set to a non-zero value."
        )

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )
    logger.debug(f"Initializing {args.device}")

    tokenizer, model = init_model(args.model_name_or_path, device)
    if not args.WT5:
        examples = load_data_generative(args.in_file, args.task)
    else:
        examples = load_data(args.in_file, args.WT5, task=args.task)
    # examples = [i[:2] for i in examples] # 3rd element (if any) is the input ids

    logger.info(examples[:5])

    special_tokens = ["[premise]", "[hypothesis]", "[update_type]", "<intensifier>", "<attenuator>", "<eos>", "[update]", "[rationale]", "[update_type_rationale]", "[update_rationale]","[update_type_no_rationale]", "[multi_no_rationale]", "[update_no_rationale]"]
#    ["[premise]", "[hypo]", "[intensifier]", "[attenuator]"]

    generate = (
        generate_conditional
        if "t5" in args.model_name_or_path or "bart" in args.model_name_or_path
        else generate_regular
    )

    with open(args.out_file, "w") as f_out:
#        for input, output, gid in tqdm.tqdm(examples):
        for input, output in tqdm.tqdm(examples):
            try:
                preds = generate(
                    tokenizer,
                    model,
                    args,
                    input,
                    device,
                )

                # For some reason some special tokens are still predicted
#                for special_token in special_tokens:
#                    preds = [pred.replace(special_token, "") for pred in preds]

                if args.task in ["update_rationale", "update_type_rationale"]:
                    preds = [pred.split(" [rationale] ")[1] if " [rationale] " in pred else pred for pred in preds]

                # Remove any word that has "]" or "[" in it
                preds = [re.sub(r"(\w*\])", "", pred) for pred in preds]
                preds = [re.sub(r"(\[\w*)", "", pred) for pred in preds]
                preds = [re.sub(" +", " ", pred).strip() for pred in preds]

            except Exception as exp:
                logger.info(exp)
                preds = []

            f_out.write(
                json.dumps({"input": input, "gold": output, "predictions": preds})
                + "\n"
            )
#            json.dumps({"gid": gid, "input": input, "gold": output, "predictions": preds})


def generate_conditional(tokenizer, model, args, input, device):
    """
    Generate a sequence with models like Bart and T5
    """
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
    decoder_start_token_id = input_ids[-1]
    input_ids = torch.tensor([input_ids]).to(device)
    max_length = args.max_length

    # Faeze added
#    stop_token = tokenizer.convert_tokens_to_ids("<eos>")

    outputs = model.generate(
        input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        min_length=5,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=decoder_start_token_id,
        num_return_sequences=1 #max(1, args.beams)
    )


    preds = [tokenizer.decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=False) for output in outputs]
    # Faeze added to remove last incomplte sentence
#    preds = [" ".join(pred.split(".", -1)[:-1]) for pred in preds]

    return preds


def generate_regular(tokenizer, model, args, input, device):
    """
    Generate a sequence with models like GPT, GPT2, or XLNet
    """
    context_tokens = tokenizer.encode(input)
    max_length = args.max_length + len(context_tokens)
    input_ids = torch.tensor(context_tokens, device=device).unsqueeze(0)

    outputs = model.generate(
        input_ids=input_ids,
        do_sample=args.beams == 0,
        max_length=max_length,
        temperature=args.temperature,
        top_p=args.p if args.p > 0 else None,
        top_k=args.k if args.k > 0 else None,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=args.beams if args.beams > 0 else None,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=1 #max(1, args.beams)
    )

    preds = [tokenizer.decode(output, skip_special_tokens=True)[len(input):].strip() for output in outputs]
#    preds = [pred.split(".")[0] for pred in preds]

    return preds


if __name__ == "__main__":
    main()
