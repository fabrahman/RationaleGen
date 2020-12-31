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
# HF transformers glue modified by Faeze Brahman
""" GLUE processors and helpers """

import logging
import os
import copy
import json

logger = logging.getLogger(__name__)


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def is_punctuation(c):
    if c == '.' or c == "," or c == "`" or c == '"' or c == "'" or c == '(' or c == ')' or c == '-' or c == '' or c == '*' or c == "/":
        return True
    return False

def get_tokens(example):
    """
    convert an example string into tokens
    Args:
        example: A single string example.
    Return:
        List of tokens.
    """
    doc_tokens = []
    prev_is_whitespace, prev_is_punc = True, True
    for c in example:
        if is_punctuation(c):
            prev_is_punc = True
            doc_tokens.append(c)
        elif is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace or prev_is_punc:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
            prev_is_punc = False
    return doc_tokens

def get_subtoken_map(subtoken_list, token_list):
    subtoken_list = [w.strip("Ġ") for w in subtoken_list]
    token_list = [w.lower() for w in token_list]
#    print(subtoken_list)
#    print(token_list)
    subtoken_map = []
    word_ind = 0
    curr = ""
    saw_q = False # if reach quatation
    for st in subtoken_list:
        subtoken_map.append(word_ind)
        if st == 'âĢ':
            continue
        if st == 'ľ':
            st = "“"
        if st == 'Ŀ':
            st = "”"
        if st == 'ĺ':
            st = '‘'
        if st == 'Ļ':
            st = '’'

        if st not in token_list[word_ind]:
            logger.info("sub-token not found in full-token ...")
            break
            
        curr += st
        if curr == token_list[word_ind]:
            word_ind += 1
            curr = ""

    return subtoken_map


def glue_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        len_examples = len(examples)
#        if ex_index % 10000 == 0:
#            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # Faeze <----!
        input_tokens = "<s> " + example.text_a + "</s></s> " + example.text_b + " </s>"
        input_tokens = input_tokens.split()
        subtoken_map = get_subtoken_map(tokenizer.convert_ids_to_tokens(input_ids), input_tokens)
        subtoken_map[0], subtoken_map[-1] = -1, -1
        assert len(input_ids) == len(subtoken_map), "Error with input length {} vs subtoken map length {}".format(len(input_ids), len(subtoken_map))

        # Faeze !---->

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            # Faeze <----!
            subtoken_map = subtoken_map + ([0] * padding_length)
            # Faeze !---->

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        # Faeze <----!
        assert len(subtoken_map) == max_length, "Error with input length {} vs {}".format(len(subtoken_map), max_length)
        # Faeze !---->

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))
            logger.info("subtoken_map: %s" % " ".join([str(x) for x in subtoken_map]))
        
        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label, input_tokens=input_tokens, subtoken_map=subtoken_map
            )
        )

    return features


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None, input_tokens=None, subtoken_map=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.input_tokens = input_tokens
        self.subtoken_map = subtoken_map

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# glue_tasks_num_labels = {
#     "cola": 2,
#     "mnli": 3,
#     "mrpc": 2,
#     "sst-2": 2,
#     "sts-b": 1,
#     "qqp": 2,
#     "qnli": 2,
#     "rte": 2,
#     "wnli": 2,
# }
#
# glue_processors = {
#     "cola": ColaProcessor,
#     "mnli": MnliProcessor,
#     "mnli-mm": MnliMismatchedProcessor,
#     "mrpc": MrpcProcessor,
#     "sst-2": Sst2Processor,
#     "sts-b": StsbProcessor,
#     "qqp": QqpProcessor,
#     "qnli": QnliProcessor,
#     "rte": RteProcessor,
#     "wnli": WnliProcessor,
# }
#
# glue_output_modes = {
#     "cola": "classification",
#     "mnli": "classification",
#     "mnli-mm": "classification",
#     "mrpc": "classification",
#     "sst-2": "classification",
#     "sts-b": "regression",
#     "qqp": "classification",
#     "qnli": "classification",
#     "rte": "classification",
#     "wnli": "classification",
# }
