import json
import os
from logging import info

import stanza
from tqdm import tqdm
from data_processing.parse import read_squad_rewrites, get_squad_question_to_answers_map
from data_processing.pre_processing import DataPreprocessor
from defs import REPEAT_Q_RAW_DATASETS, SQUAD_REWRITE_MTURK_DIR, SQUAD_REWRITES_SYNTHETIC_JSON


def generate_repeat_q_squad_raw(use_triples: bool, mapped_triples: bool):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,ner')

    question_to_answers_map = get_squad_question_to_answers_map()

    def _get_tokens(words):
        return " ".join([w.text.lower() for w in words])

    def _get_pos_sequence(words):
        return " ".join([w.xpos for w in words])

    def _get_tags(words, sought_after_tokens, beg_tag, inside_tag, tag_list=None):
        if tag_list is None:
            tags = ["O" for _ in range(len(words))]
        else:
            tags = tag_list
        for i in range(len(words)):
            if words[i].text == sought_after_tokens[0]:
                complete_match = True
                for j in range(len(sought_after_tokens)):
                    if i+j >= len(words) or sought_after_tokens[j] != words[i+j].text:
                        complete_match = False
                        break
                if complete_match:
                    tags[i] = beg_tag
                    for j in range(i+1, i+len(sought_after_tokens)):
                        tags[j] = inside_tag
        return tags

    def _get_entity_tags(question_doc, answers, facts_sentences):
        q_entities = [ent.text for ent in question_doc.entities]
        q_ent_tags = ["O" for _ in range(question_doc.num_words)]
        facts_ent_tags = [["O" for _ in range(len(facts_sentences[i].words))] for i in range(len(facts_sentences))]
        for q_entity in q_entities:
            q_entity_toks = q_entity.split()
            # Marks named entities in question
            q_ent_tags = _get_tags(list(question_doc.iter_words()), q_entity_toks, beg_tag="BN", inside_tag="IN",
                                   tag_list=q_ent_tags)
            # Marks NEs from the question in facts
            facts_ent_tags = [_get_tags(facts_sentences[i].words, q_entity_toks, "BN", "IN", facts_ent_tags[i])
                              for i in range(len(facts_sentences))]
        # Overwrites tags with answer tags if any in facts
        for answer in answers:
            answer_tokens = answer.lower().split()
            facts_ent_tags = [_get_tags(facts_sentences[i].words, answer_tokens, "BA", "IA", facts_ent_tags[i])
                              for i in range(len(facts_sentences))]
        return " ".join(q_ent_tags), [" ".join(f) for f in facts_ent_tags]

    def _get_cases(words):
        return " ".join(["UP" if word.text[0].isupper() else "LOW" for word in words])

    def _make_example(_base_question, _rewritten_question, _facts, _answers, _is_synthetic):
        try:
            # Create example placeholders and filter out irrelevant question words for future word matching
            analyzed_question = nlp(_base_question)
            analyzed_target = nlp(_rewritten_question)
            if use_triples:
                if mapped_triples:
                    analyzed_facts = [nlp(triple).sentences[0] for triple in _facts]
                else:
                    analyzed_facts = [nlp(triple).sentences[0] for fact in _facts for triple in fact]
            else:
                analyzed_facts = [fact_sentence for fact in _facts for fact_sentence in nlp(fact).sentences]
            base_question_entity_tags, facts_entity_tags = _get_entity_tags(analyzed_question, _answers, analyzed_facts)
            question_words = list(analyzed_question.iter_words())
            example = {
                "base_question": _get_tokens(question_words),
                "base_question_pos_tags": _get_pos_sequence(question_words),
                "base_question_entity_tags": base_question_entity_tags,
                "base_question_letter_cases": _get_cases(question_words),
                "base_question_ner": DataPreprocessor.create_ner_sequence(True, question_words),
                "facts": [_get_tokens(fact.words) for fact in analyzed_facts],
                "facts_entity_tags": facts_entity_tags,
                "facts_pos_tags": [_get_pos_sequence(fact.words) for fact in analyzed_facts],
                "facts_letter_cases": [_get_cases(sentence.words) for sentence in analyzed_facts],
                "facts_ner": [DataPreprocessor.create_ner_sequence(True, fact.words)
                              for fact in analyzed_facts],
                "target": _get_tokens(analyzed_target.iter_words()),
                "is_synthetic": _is_synthetic
            }
            return example
        except Exception as e:
            print(e)
            print("Question:")
            print(_base_question)
            print("Target:")
            print(_rewritten_question)
            print("Facts:")
            [print(f) for f in _facts]
        return None

    for mode in ("test", "dev", "train"):
        ds = []
        orga_filepath = f"{SQUAD_REWRITE_MTURK_DIR}/{mode}.json"
        examples = {
            "organic": read_squad_rewrites(
                dataset_path=orga_filepath, use_triples=use_triples, mapped_triples=mapped_triples
            ),
            "synthetic": read_squad_rewrites(
               SQUAD_REWRITES_SYNTHETIC_JSON, use_triples=use_triples, mapped_triples=mapped_triples
            ) if mode == "train" else []
        }

        for data_type in ("organic", "synthetic"):
            for example in tqdm(examples[data_type]):
                answers = question_to_answers_map[example["base_question"].strip()]
                ex = _make_example(
                    _base_question=example["base_question"],
                    _rewritten_question=example["target"],
                    _facts=example["facts"],
                    _answers=answers,
                    _is_synthetic=data_type == "synthetic"
                )
                if ex is not None:
                    ds.append(ex)

        if not os.path.exists(REPEAT_Q_RAW_DATASETS):
            os.mkdir(REPEAT_Q_RAW_DATASETS)
        ds_filename = f"{REPEAT_Q_RAW_DATASETS}/squad"
        if mapped_triples:
            ds_filename = f"{ds_filename}_mapped"
        if use_triples:
            ds_filename = f"{ds_filename}_triples"
        ds_filename = f"{ds_filename}_{mode}.json"
        with open(ds_filename, mode='w') as f:
            json.dump(ds, f, indent=4)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, choices=("squad_repeat_q_triples", "squad_repeat_q_mapped_triples",
                                                           "squad_repeat_q"))
    args = parser.parse_args()

    if "squad_repeat_q" in args.dataset_name:
        use_triples = False
        mapped_triples = False
        if "triples" in args.dataset_name:
            use_triples = True
        if "mapped" in args.dataset_name:
            mapped_triples = True
        generate_repeat_q_squad_raw(use_triples=use_triples, mapped_triples=mapped_triples)
        info(f"Raw SQuAD dataset for {args.dataset_name} generated.")
    else:
        raise ValueError("Non-existing dataset type")
    print("Done")
