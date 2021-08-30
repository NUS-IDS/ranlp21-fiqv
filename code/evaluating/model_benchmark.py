import json

from evaluating.rouge_score import rouge_l_sentence_level as rouge_l
import nltk.translate.bleu_score as bleu
from nltk.translate.meteor_score import meteor_score as meteor
from nltk.metrics import scores
import pandas as pd
import numpy as np
from defs import REPEAT_Q_SQUAD_DATA_DIR, REPEAT_Q_SQUAD_OUTPUT_FILEPATH


def corpus_f1_score(corpus_candidates, corpus_references):
    def f1_max(candidate, references):
        f1 = 0.0
        for ref in references:
            f1 = max(f1, scores.f_measure(set(ref), set(candidate)))
        return f1

    return np.mean(np.array([f1_max(candidate, references)
                             for (references, candidate) in zip(corpus_references, corpus_candidates)]))


def benchmark(corpus_candidates: np.ndarray, corpus_references: np.ndarray):
    corpus_candidates_split = [candidate.strip().split(' ') for candidate in corpus_candidates]
    corpus_references_split = [[reference.strip().split(' ') for reference in refs] for refs in corpus_references]
    bleu_1 = 100 * bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(1.0,))
    print(f"BLEU-1: {bleu_1}")
    bleu_2 = 100 * bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(0.5, 0.5))
    print(f"BLEU-2: {bleu_2}")
    bleu_3 = 100 * bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(1.0 / 3, 1.0 / 3, 1.0 / 3))
    print(f"BLEU-3: {bleu_3}")
    bleu_4 = 100 * bleu.corpus_bleu(corpus_references_split, corpus_candidates_split, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-4: {bleu_4}")
    # Sentences level ROUGE-L with beta = P_lcs / (R_lcs + 1e-12)
    rouge_l_sentence_level = 100 * rouge_l(corpus_candidates_split, corpus_references_split)
    print(f"ROUGE-L: {rouge_l_sentence_level}")
    meteor_score = 100 * np.mean(np.array([meteor(references, candidate)
                                     for (references, candidate) in zip(corpus_references, corpus_candidates)]))
    print(f"METEOR macro average: {meteor_score}")
    f1_score = 100 * corpus_f1_score(corpus_candidates_split, corpus_references_split)
    print(f"F1 macro average: {f1_score}")


def prepare_for_eval(preds: pd.DataFrame, targets: pd.DataFrame, test_passages: pd.DataFrame,
                     train_passages: pd.DataFrame):
    corpus_candidates = {}
    corpus_references = {}

    for candidate, reference, source in zip(preds.values, targets.values, test_passages.values):
        # Ignores passages that were already contained in the training set
        candidate = candidate[0]
        reference = reference[0]
        source = source[0]
        if not train_passages[train_passages.columns[0]].str.contains(source, regex=False).any():
            if source in corpus_references:
                corpus_references[source].append(reference)
                if len(candidate) > len(corpus_candidates[source]):
                    corpus_candidates[source] = candidate
            else:
                corpus_references[source] = [reference]
                corpus_candidates[source] = candidate

    corpus_references = corpus_references.values()
    corpus_candidates = corpus_candidates.values()
    assert len(corpus_candidates) == len(corpus_references)
    return corpus_candidates, corpus_references


def get_sg_dqg_predictions(pred_path):
    preds = []
    refs = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("<gold>"):
                refs.append([line[len("<gold>\t"):]])
            elif line.startswith("<pred>"):
                preds.append(line[len("<pred>\t"):])
    return preds, refs


if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--collate_ngrams', action='store_true', help='Removes n-gram duplicates.')

    args = parser.parse_args()

    model = args.model_name
    candidates, references = {}, {}

    with open(f"{REPEAT_Q_SQUAD_DATA_DIR}/test.data.json", mode='r') as test_file:
        test_data = json.load(test_file)

    for data in test_data:
        q = data["base_question"]
        # Only keeps organic data for evaluation
        if data["passage_id"] == -1:
            rewrites = references.get(q)
            if rewrites is None:
                rewrites = [data["target"]]
            else:
                rewrites.append(data["target"])
            references[q] = rewrites

    base_questions = [k for k, _ in references.items()]
    references = [v for _, v in references.items()]
    candidates = np.array(pd.read_csv(
        REPEAT_Q_SQUAD_OUTPUT_FILEPATH, header=None, sep='\n', comment=None)
    ).reshape((-1,))
    # Keeps insertion order and gets rid of duplicates
    candidates = list({c: None for c in candidates}.keys())

    benchmark(candidates, references)
