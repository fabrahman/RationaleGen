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

# Modified by Rachel Rudinger 2020 / Faeze Brahman 2020
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
from typing import Optional, List, Dict

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

class DefeasibleClassifierProcessor(DataProcessor):
    """Processor for the Defeasible Inference data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir, eval_on="dev"):
        """See base class."""
        logger.info("LOOKING AT {} {}".format(data_dir, eval_on))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "{}.csv".format(eval_on))), eval_on)

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
        return ["0", "1"]

    def _read_csv(self, input_file):
        """Reads a comma separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter=",", quotechar='"'))

    def _create_examples(self, lines: List[List[str]],  type: str):
        """Creates examples for the training and dev sets."""
        
        examples = []
        for line in lines[1:]:
#            # skip cases where strengtheners or weakeners are impossible
#            #if len(line) < 14:
#            #    print(line)
#            #    print("LEN: "+str(len(line)))
#            assert len(line) == 14
#            if line[8] == "on" or line[11] == "on":
#                continue
#            example_id = ":".join([line[0],line[6]])
#            text_a = (line[4]+" "+line[5]).replace("é", "e").replace("ñ", "n") # prem-hyp pair
##            if exp:
##                text_a += " " + line[14] # add explanation if available
#            text_b_w = line[7].replace("é", "e").replace("ñ", "n") # weakener
#            text_b_s = line[10].replace("é", "e").replace("ñ", "n") # strengthener
#            # Weakener
#            examples.append(InputExample(guid=example_id+":W", text_a=text_a, text_b=text_b_w, label=str(0)))
#            # Strengthener
#            examples.append(InputExample(guid=example_id+":S", text_a=text_a, text_b=text_b_s, label=str(1)))

            # skip cases where strengtheners or weakeners are impossible
            #if len(line) < 14:
            #    print(line)
            #    print("LEN: "+str(len(line)))
            assert len(line) == 14
            if line[8] == "on" or line[11] == "on":
                continue
            example_id = ":".join([line[0],line[6]])
            prem = line[4].strip(".") + "."
            weakener = line[7].strip(".") + "."
            strengthener = line[10].strip(".") + "."
            hypo = line[5].strip(".") + "."

            text_a_w = (prem  + " "+ weakener).replace("é", "e").replace("ñ", "n") # prem-upd pair
            text_a_s = (prem + " " + strengthener).replace("é", "e").replace("ñ", "n")
#            if exp:
#                text_a += " " + line[14] # add explanation if available
            text_b = hypo.replace("é", "e").replace("ñ", "n") # hypothesis
            # Weakener
            examples.append(InputExample(guid=example_id+":W", text_a=text_a_w, text_b=text_b, label=str(0), prem=prem, hypo=hypo, upd=weakener ))
            # Strengthener
            examples.append(InputExample(guid=example_id+":S", text_a=text_a_s, text_b=text_b, label=str(1), prem=prem, hypo=hypo, upd=strengthener))
        return examples


class DefeasibleClassifierHypOnlyProcessor(DefeasibleClassifierProcessor):
    """Processor for the Defeasible Inference data set."""

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
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


class eSNLIClassifierProcessor(DataProcessor):
    """Processor for the eSNLI Inference data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir, eval_on="dev"):
        """See base class."""
        logger.info("LOOKING AT {} {}".format(data_dir, eval_on))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "{}.csv".format(eval_on))), eval_on)

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _read_csv(self, input_file):
        """Reads a comma separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter=",", quotechar='"'))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""

        examples = []
        for line in lines[1:]:

#            assert len(line) == 11
            example_id = line[1]
            text_a = line[3] + " [SEP] " + line[5]
            text_b = line[4]
            label = "1" if line[2] == "entailment" else "0"

            if text_a == "" or text_b == "":
                continue

            examples.append(
                InputExample(guid=example_id , text_a=text_a, text_b=text_b, label=label))

        return examples


class eSNLIPrecitInstanceProcessor:
    """Processor for instance-wise prediction for esnli classifier tested on definf with distant-supervised rationales"""

    def get_labels(self):
        """See base class."""
        # return ["0", "1", "2", "3"]
        return ["0", "1"]

    def get_examples(self, line):
        examples = []
        assert len(line) == 10

        example_id = line["annotation_id"]
        text_pref = line["premise"] + " " + line["update"]
        text_b = line["hypothesis"]
        label = "1" if example_id[-1] == "S" else "0"

        for item in line["rationales"]:
            text_a = text_pref + " [SEP] " + item
            examples.append(
                InputExample(guid=example_id, text_a=text_a, text_b=text_b, label=label))

        return examples
