import ast
import json
import pandas as pd
from tqdm import tqdm
from defs import SQUAD_TRAIN, SQUAD_DEV
import os


def next_chunk(file_reader):
    next_content = None
    next_line = file_reader.readline()
    while next_line and next_line == "\n":
        next_line = file_reader.readline()
    while next_line and next_line != "\n":
        next_line = next_line.strip()
        if next_content is None:
            next_content = [next_line]
        else:
            next_content.append(next_line)
        next_line = file_reader.readline()
    return next_content, next_line


def read_squad_facts_files(facts_dirpath):
    fact_dataset = {}
    assert os.path.isdir(facts_dirpath)
    print(f"Parsing {facts_dirpath}...")
    for filename in tqdm(os.listdir(facts_dirpath)):
        if "facts" in filename:
            passage_facts = []
            passage_id = int(filename.replace("facts.", "").replace(".txt", ""))
            with open(os.path.join(facts_dirpath, filename), mode='r') as f:
                last_line = ""
                while last_line is not None:
                    next_content, last_line = next_chunk(f)
                    if next_content is None:
                        break
                    fact_id = next_content[0]
                    g_type = next_content[1]
                    g_name = next_content[2]
                    g_description = next_content[3]
                    g_article_text = next_content[4]

                    fields_with_keys_and_renamed_keys = zip(
                        [fact_id, g_type, g_name, g_description, g_article_text],
                        ["FACTID", "GKGTYPE", "GKGNAME", "GKGDESC", "GKGARTTEXT"],
                        ["fact_id", "type", "name", "description", "text"]
                    )
                    fact = {}
                    for field, key, renamed_key in fields_with_keys_and_renamed_keys:
                        assert key in field
                        fact[renamed_key] = field.replace(key, "").strip()
                    passage_facts.append(fact)

            fact_dataset[passage_id] = passage_facts
    return fact_dataset


def read_squad_rewrites_files(rewrites_dirpath):
    assert os.path.isdir(rewrites_dirpath)
    rewrites = {}
    print(f"Parsing {rewrites_dirpath}...")
    for filename in tqdm(os.listdir(rewrites_dirpath)):
        if "qw" in filename and "old" not in filename:
            passage_rewrites = []
            passage_id = int(filename.replace("qw.", "").replace(".list", ""))
            with open(os.path.join(rewrites_dirpath, filename), mode='r') as f:
                next_line = ""
                while next_line is not None:
                    next_content, next_line = next_chunk(f)
                    if next_content is None:
                        break
                    base_question = next_content[0]
                    # Some questions have more than one rewrites, we create one example per rewrite
                    rewritten_questions = next_content[1:]
                    for rewritten_question in rewritten_questions:
                        passage_rewrites.append({
                            "base_question": base_question,
                            "rephrased": rewritten_question
                        })
            rewrites[passage_id] = passage_rewrites
    return rewrites


def read_squad_qmap_files(qmap_dirpath):
    assert os.path.isdir(qmap_dirpath)
    question_to_facts = {}
    for filename in tqdm(os.listdir(qmap_dirpath)):
        if "qmap" in filename:
            passage_id = int(filename.replace("qmap.", "").replace(".txt", ""))
            q_maps = {}
            with open(os.path.join(qmap_dirpath, filename), mode='r') as f:
                next_line = ""
                while next_line is not None:
                    next_content, next_line = next_chunk(f)
                    if next_content is None:
                        break
                    question = next_content[0].replace("QUESTION ", "")
                    fact_ids = ast.literal_eval(next_content[1].replace("facts ", ""))
                    q_maps[question] = fact_ids
            question_to_facts[passage_id] = q_maps
    return question_to_facts


def read_squad_rewrites(dataset_path, use_triples=False, mapped_triples=False):
    with open(dataset_path, mode="r") as f:
        data = json.load(f)
        questions_to_facts = {}
        questions_to_rewrites = {}
        ds = []

        if mapped_triples:
            for datapoint in data:
                for rf_pair in datapoint["rfpairs"]:
                    ds.append({
                        "base_question": datapoint["question"],
                        "facts": [rf_pair["triple"]],
                        "target": rf_pair["rewrite"],
                        "passage_id": -1
                    })
        else:
            for datapoint in data:
                q = datapoint["question"]
                if use_triples:
                    triples = [rfpair["triple"] for rfpair in datapoint["rfpairs"]]
                    fact = " ".join(triples)
                else:
                    fact = datapoint["fact"]
                rewrites = [rfpair["rewrite"] for rfpair in datapoint["rfpairs"]]
                if q in questions_to_facts:
                    questions_to_facts[q].append(fact)
                    questions_to_rewrites[q].extend(rewrites)
                else:
                    questions_to_facts[q] = [fact]
                    questions_to_rewrites[q] = rewrites
            for q in questions_to_facts.keys():
                for target in questions_to_rewrites[q]:
                    ds.append({
                        "base_question": q,
                        "facts": questions_to_facts[q],
                        "target": target
                    })
    return ds


def get_squad_question_to_answers_map():
    qa_map = {}

    def _read_qas(path):
        ds = pd.read_json(path)["data"]
        for article in ds:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    qa_map[qa["question"].strip()] = [answer["text"] for answer in qa["answers"]]
    _read_qas(SQUAD_DEV)
    _read_qas(SQUAD_TRAIN)
    return qa_map
