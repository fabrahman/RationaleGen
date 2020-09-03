import argparse
import numpy as np
import pandas as pd

from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', action='append', type=str, required=True, help="batch results file paths")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory, where the script saves the weakener.csv and strengthener.csv files")
    args = parser.parse_args()
    
    dfs = [pd.read_csv(file) for file in args.file]
    df = dfs[0] if len(dfs) == 1 else pd.concat(dfs)

    # Group by inputs (HITId, premise, hypothesis, update + type, rationales, sources)
    grouped = df.groupby(["HITId"] + [col for col in df.columns if "Input" in col])[[col for col in df.columns if "Answer" in col]].agg(list).reset_index()

    # Get the most frequent value for each column
    majority = grouped.copy()

    for _, row in majority.iterrows(): 
        for i, col in enumerate(majority.columns): 
            if "Answer" in col:
                row[i] = Counter(row[i]).most_common(1)[0][0]

    # Ignore rationale numbers
    update_types = ["weakener", "strengthener"]
    reduced = pd.concat([majority[[col for col in majority.columns if f"rationale{idx}" in col]].rename(columns={col: col.replace(f"_rationale{idx}", "") for col in majority.columns if f"rationale{idx}" in col}) for idx in range(1, 7)])
    by_update_type = {update_type: reduced[[col for col in reduced.columns if update_type in col]].rename(columns={col: col.replace(f"{update_type}_", "") for col in reduced.columns if update_type in col}) for update_type in update_types}

    # Group by source and update type and compute percents
    percents = {}
    props = ["relevant", "correct", "explains", "gibberish_understandable_grammatical"]
    grammar_props = {"gibberish", "understandable", "grammatical"}

    for update_type, df in by_update_type.items():
        prop_cols = [col for col in df.columns if any([prop in col for prop in props])]
        curr = df.groupby("Input.method")[prop_cols].agg(list).reset_index()
        
        overall_row = ["Overall"] + [np.concatenate(curr[col].values).tolist() for col in curr.columns[1:]]
        overall_row = pd.DataFrame([overall_row], columns=curr.columns)
        curr = curr.append(overall_row, sort=False)

        for prop in prop_cols:
            curr[prop] = curr[prop].apply(lambda lst: Counter(lst))

            if "grammatical" not in prop:
                curr[prop] = curr[prop].apply(lambda ctr: ctr["on"] * 100.0 / sum(ctr.values()))
            else:
                for val in grammar_props:
                    curr[val] = curr[prop].apply(lambda ctr: ctr.get(val, 0) * 100.0 / sum(ctr.values()))
                curr = curr.drop(columns=["Answer.gibberish_understandable_grammatical"])
                curr["grammatical_or_understanable"] = curr["grammatical"] + curr["understandable"]

        percents[update_type] = curr
    
    # Save 
    for update_type in update_types:
        percents[update_type].to_csv(f"{args.out_dir}/{update_type}.csv")
    

if __name__ == "__main__":
    main()
