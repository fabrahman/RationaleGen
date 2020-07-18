# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Rachel Rudinger 2020
""" GLUE processors and helpers """

# utils_multiple_choice imports
import csv
import glob
import json
import logging
import os
import random as rand
import itertools
from typing import List

import tqdm

#####

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)




import logging
import os

#from ...file_utils import is_tf_available
from utils import DataProcessor, InputExample, InputFeatures
from typing import List

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class DefeasibleClassifierProcessor(DataProcessor):
    """Processor for the Defeasible Inference data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir, eval_on="dev"):
        """See base class."""
        logger.info("LOOKING AT {} {}".format(data_dir, eval_on))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "{}.csv".format(eval_on))))

    #def get_test_examples(self, data_dir):
    #    """See base class."""
    #    logger.info("LOOKING AT {} dev".format(data_dir))
    #    raise ValueError(
    #        "For swag testing, the input file does not contain a label column. It can not be tested in current code"
    #        "setting!"
    #    )
    #    return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        #return ["0", "1", "2", "3"]
        return ["0", "1"]

    def _read_csv(self, input_file):
        #with open(input_file, "r", encoding="utf-8") as f:
        #    return list(csv.reader(f))
        """Reads a comma separated value file."""
        #with open(input_file, "r", encoding="utf-8-sig") as f:
        #with open(input_file, "r", encoding="utf-8", newline='\n') as f:
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter=",", quotechar='"'))

    def _create_examples(self, lines: List[List[str]],  type: str):
        """Creates examples for the training and dev sets."""
        
        examples = []
        for line in lines[1:]:
            # skip cases where strengtheners or weakeners are impossible
            #if len(line) < 14:
            #    print(line)
            #    print("LEN: "+str(len(line)))
            assert len(line) == 14
            if line[8] == "on" or line[11] == "on":
                continue
            example_id = ":".join([line[0],line[6]])
            text_a = line[4]+" "+line[5] # prem-hyp pair
#            if exp:
#                text_a += " " + line[14] # add explanation if available
            text_b_w = line[7] # weakener
            text_b_s = line[10] # strengthener
            # Weakener
            examples.append(InputExample(guid=example_id+":W", text_a=text_a, text_b=text_b_w, label=str(0)))
            # Strengthener
            examples.append(InputExample(guid=example_id+":S", text_a=text_a, text_b=text_b_s, label=str(1)))
        return examples

#0 AssignmentId
#1 WorkerId
#2 WorkTimeInSeconds
#3 Answer_feedback
#4 Input_premise
#5 Input_hypothesis
#6 Input_pairID
#7 Answer_Attenuator_modifier
#8 Answer_Attenuator_impossible
#9 Answer_Attenuator_reason
#10 Answer_Intensifier_modifier
#11 Answer_Intensifier_impossible
#12 Answer_Intensifier_reason
#13 split


class DefeasibleClassifierHypOnlyProcessor(DefeasibleClassifierProcessor):
    """Processor for the Defeasible Inference data set."""

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        #if type == "train" and lines[0][-1] != "label":
        #    raise ValueError("For training, the input file must contain a label column.")

        #examples = [
        #    InputExample(
        #        example_id=line[2],
        #        question=line[5],  # in the swag dataset, the
        #        # common beginning of each
        #        # choice is stored in "sent2".
        #        contexts=[line[4], line[4], line[4], line[4]],
        #        endings=[line[7], line[8], line[9], line[10]],
        #        label=line[11],
        #    )
        #    for line in lines[1:]  # we skip the line with the column names
        #]
        examples = []
        for line in lines[1:]:
            # skip cases where strengtheners or weakeners are impossible
            if len(line) < 13:
                print(line)
            if line[8] == "on" or line[11] == "on":
                continue
            example_id = ":".join([line[0],line[6]])
            question = ""
            #contexts = 2*[line[4]+" "+line[5]]
            contexts = 2*["<s>"] # hyp-only/context-free baseline, removes premise-hypothesis pair
            r = rand.randint(0,1)
            if r == 0:
                endings = [line[7], line[10]]
            else:
                assert r == 1
                endings = [line[10], line[7]]
            label = str(r)
            examples.append(InputExample(example_id=example_id, question=question, contexts=contexts, endings=endings, label=label))
        return examples


# Faeze added
class DefeasibleClassifierWithRationaleProcessor(DataProcessor):
    """Processor for the Defeasible Inference data set."""

    def get_train_examples(self, data_dir, exp_from):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_jsonline(os.path.join(data_dir, "train_rationalized_{}.jsonl".format(exp_from))), "train") # "definf_train_predictions_beam5.jsonl"

    def get_dev_examples(self, data_dir, exp_from, eval_on="dev"):
        """See base class."""
        logger.info("LOOKING AT {} {}".format(data_dir, eval_on))
        return self._create_examples(self._read_jsonline(os.path.join(data_dir, "{}_rationalized_{}.jsonl".format(eval_on, exp_from))), eval_on) #"definf_{}_predictions_beam5.jsonl

    # def get_test_examples(self, data_dir):
    #    """See base class."""
    #    logger.info("LOOKING AT {} dev".format(data_dir))
    #    raise ValueError(
    #        "For swag testing, the input file does not contain a label column. It can not be tested in current code"
    #        "setting!"
    #    )
    #    return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_jsonline(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        # return ["0", "1", "2", "3"]
        return ["0", "1"]

    def _read_jsonline(self, input_file):
        # with open(input_file, "r", encoding="utf-8") as f:
        #    return list(csv.reader(f))
        """Reads a jsonline file with json in each line."""
        # with open(input_file, "r", encoding="utf-8-sig") as f:
        # with open(input_file, "r", encoding="utf-8", newline='\n') as f:
        all_data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                all_data.append(json.loads(line))
        return all_data

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""

        examples = []
        for line in lines:
#            assert len(line) == 4

            example_id = line["gid"] if "gid" in line else "nan"
            prem = line["input"].split(": ", 1)[1].split(".", 1)[0] + "."
            upd_hyp = line["input"].split(": ", 1)[1].split(".", 1)[1]
            hyp = upd_hyp.split(": ")[1]
            upd = upd_hyp.split(".", 1)[0]+ "."
            rationale = line["predictions"][0].split(": ")[1].rsplit(".", 1)[0]+"."

            text_a = prem + " " + hyp + " " + rationale  # prem-hyp-rationale pair
            text_b_w = upd  # weakener
            # text_b_s = line[10]  # strengthener
            # Weakener & strengthener
            label = str(int(line["gold"].split()[0] == "intensifier"))
            examples.append(InputExample(guid=example_id , text_a=text_a, text_b=text_b_w, label=label))
            # Strengthener
            # examples.append(InputExample(guid=example_id + ":S", text_a=text_a, text_b=text_b_s, label=str(1)))
        return examples


# Faeze added
class DefeasibleClassifierRationaleFromEsnliProcessor(DefeasibleClassifierWithRationaleProcessor):
    """Processor for the Defeasible Inference data set."""

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""

        examples = []
        for line in lines:
#            assert len(line) == 4

            example_id = line["gid"] if "gid" in line else "nan"
            rationale = line["predictions"][0].split(": ")[1].rsplit(".", 1)[0]+"."

            text_a = rationale  # prem-hyp-rationale pair
            # text_b_w = upd  # weakener
            # text_b_s = line[10]  # strengthener
            # Weakener & strengthener
            label = str(int(line["gold"].split()[0] == "intensifier"))
            examples.append(InputExample(guid=example_id , text_a=text_a, text_b=None, label=label))
            # Strengthener
            # examples.append(InputExample(guid=example_id + ":S", text_a=text_a, text_b=text_b_s, label=str(1)))
        return examples

# Faeze added
class DefeasibleClassifierRationaleFromComet(DefeasibleClassifierWithRationaleProcessor):
    """Processor for the Defeasible Inference data set."""

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""

        examples = []
        for line in lines:
#            assert len(line) == 4
            if line["Answer_Attenuator_impossible"] == "on" or line["Answer_Intensifier_impossible"] == "on":
                continue

            example_id = ":".join((line["AssignmentId"], line["Input_pairID"]))
            hypo_r = line["Hypothesis_comet_supervision"]
            streng_r = line["Intensifier_comet_supervision"]
            weak_r = line["Attenuator_comet_supervision"]
            #strengthener
            for h, s in list(itertools.product(hypo_r, streng_r)):
                text_a_s = h
                text_b_s = s
                examples.append(InputExample(guid=example_id + ":S", text_a=text_a_s, text_b=text_b_s, label=str(1)))
            # weakener
            for h, s in list(itertools.product(hypo_r, weak_r)):
                text_a_w = h
                text_b_w = s
                examples.append(InputExample(guid=example_id + ":W", text_a=text_a_w, text_b=text_b_w, label=str(0)))

        return examples