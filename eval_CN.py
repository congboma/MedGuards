import re
import pandas as pd
import jieba
import numpy as np
from rouge_chinese import Rouge as RougeZh
from rouge import Rouge as RougeEn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import bert_score
import bleurt.score as bleurtscore
from tabulate import tabulate

def parse_reference_file(filepath):
    reference_corrections, reference_flags, reference_sent_id = {}, {}, {}
    df = pd.read_csv(filepath)

    for _, row in df.iterrows():
        text_id = str(row['Text ID'])
        corrected_sentence = row['Corrected Sentence']
        if not isinstance(corrected_sentence, str):
            corrected_sentence = "NA" if pd.isna(corrected_sentence) else str(corrected_sentence).replace("\n", " ").replace("\r", " ").strip()
        reference_corrections[text_id] = corrected_sentence
        reference_flags[text_id] = str(row['Error Flag'])
        reference_sent_id[text_id] = str(row['Error Sentence ID'])

    return reference_corrections, reference_flags, reference_sent_id

def parse_run_submission_file(filepath):
    candidate_corrections, predicted_flags, candidate_sent_id = {}, {}, {}

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        line = re.sub('\s+', ' ', line)
        items = line.split()
        text_id, error_flag, sentence_id = items[0], items[1], items[2]
        corrected_sentence = ' '.join(items[3:]).strip()

        corrected_sentence = corrected_sentence.strip('"')

        predicted_flags[text_id] = error_flag
        candidate_sent_id[text_id] = sentence_id
        candidate_corrections[text_id] = "NA" if error_flag == '0' else corrected_sentence

    return candidate_corrections, predicted_flags, candidate_sent_id

def get_nlg_eval_data(reference_corrections, candidate_corrections):
    references, predictions, counters = [], [], []
    for text_id in reference_corrections:
        if (
            text_id in candidate_corrections and
            reference_corrections[text_id] != "NA" and
            candidate_corrections[text_id] != "NA"
        ):
            references.append(reference_corrections[text_id])
            predictions.append(candidate_corrections[text_id])
            counters.append(text_id)
    return references, predictions, counters

def compute_accuracy(reference_flags, reference_sent_id, predicted_flags, candidate_sent_id):
    matching_flags_nb = sum(
        1 for text_id in reference_flags
        if text_id in predicted_flags and reference_flags[text_id] == predicted_flags[text_id]
    )
    flags_accuracy = matching_flags_nb / len(reference_flags)

    matching_sentence_nb = sum(
        1 for text_id in reference_sent_id
        if text_id in candidate_sent_id and candidate_sent_id[text_id] == reference_sent_id[text_id]
    )
    sent_accuracy = matching_sentence_nb / len(reference_sent_id)

    return flags_accuracy, sent_accuracy

class NLGMetrics:
    def __init__(self):
        self.rouge_zh = RougeZh()
        self.rouge_en = RougeEn()
        self.smoothie = SmoothingFunction().method4

    def compute_rouge(self, references, predictions):
        rouge1f_scores, rouge2f_scores, rougeLf_scores = [], [], []
        for ref, pred in zip(references, predictions):
            ref_seg = ' '.join(jieba.cut(ref))
            pred_seg = ' '.join(jieba.cut(pred))
            # ref_seg = ' '.join(list(ref))
            # pred_seg = ' '.join(list(pred))

            scores_zh = self.rouge_zh.get_scores(pred_seg, ref_seg)[0]

            rouge1f_scores.append(scores_zh["rouge-1"]["f"])
            rouge2f_scores.append(scores_zh["rouge-2"]["f"])
            rougeLf_scores.append(scores_zh["rouge-l"]["f"])

        return (
            float(np.mean(rouge1f_scores)),
            float(np.mean(rouge2f_scores)),
            float(np.mean(rougeLf_scores))
        )

    def compute_bleu(self, references, predictions):
        bleu_scores = []
        for ref, pred in zip(references, predictions):
            # ref_tokens = ref.split()
            # pred_tokens = pred.split()
            pred_clean = pred.strip('"\n ')
            pred_tokens = list(jieba.cut(pred_clean))
            ref_tokens = list(jieba.cut(ref))
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smoothie)
            bleu_scores.append(bleu)
        return float(np.mean(bleu_scores))

    def compute_bertscore(self, references, predictions):
        _, _, bertScore_F1 = bert_score.score(predictions, references, lang="zh", verbose=True)
        # return float(np.mean(bertScore_F1.numpy()))
        return bertScore_F1

    def compute_bleurt(self, references, predictions):
        bleurt_path = "BLEURT-20"
        try:
            bleurtscorer = bleurtscore.BleurtScorer(checkpoint=bleurt_path)
            bleurtscores = bleurtscorer.score(references=references, candidates=predictions, batch_size=1)
            # return float(np.mean(bleurtscores))
            return bleurtscores
        except Exception as e:
            print(f"BLEURT Fail: {str(e)}")
            return [0.0]

if __name__ == "__main__":
    # file path
    submission_file = "XXX.txt"
    reference_csv_file = "XXX.csv"

    reference_corrections, reference_flags, reference_sent_id = parse_reference_file(reference_csv_file)
    candidate_corrections, candidate_flags, candidate_sent_id = parse_run_submission_file(submission_file)

    flags_acc, sent_acc = compute_accuracy(reference_flags, reference_sent_id, candidate_flags, candidate_sent_id)

    references, predictions, counters = get_nlg_eval_data(reference_corrections, candidate_corrections)

    metrics = NLGMetrics()
    r1, r2, rl = metrics.compute_rouge(references, predictions)
    bleu = metrics.compute_bleu(references, predictions)
    bert = metrics.compute_bertscore(references, predictions)
    bleurt = metrics.compute_bleurt(references, predictions)

    headers = ["flags.acc", "sent_acc", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore", "BLEURT"]
    data = [[flags_acc, sent_acc, r1, r2, rl, bleu, bert, bleurt]]

    print(tabulate(data, headers=headers, floatfmt=".4f", tablefmt="grid"))
