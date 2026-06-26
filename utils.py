"""
Error Correction with Multi-Agent
"""

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import Levenshtein

from rouge_score import rouge_scorer
import bert_score
import bleurt.score as bleurtscore
import numpy as np
import os
import difflib
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

from eval_CN import NLGMetrics as NLGMetrics_CN
from rouge import Rouge

import jieba
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
nltk.download('punkt')


def find_most_similar_sentence(sentences, opinion):
    if opinion == "NAN":
        return -1, "NAN"

    sentence_texts = [s.partition(' ')[2] for s in sentences]
    best_match = ''
    highest_ratio = 0.0

    ret_id = 0
    for i, text in enumerate(sentence_texts):
        ratio = difflib.SequenceMatcher(None, text, opinion).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = text
            ret_id = i

    return ret_id, best_match


def jaccard_similarity(s1, s2):
    words1 = set(jieba.lcut(s1))
    words2 = set(jieba.lcut(s2))
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def find_most_similar_sentence_CN(sentences, opinion):
    if opinion == "NAN":
        return -1, "NAN"

    sentence_texts = [s.partition(' ')[2] for s in sentences]

    best_match = ''
    highest_score = 0.0
    ret_id = -1

    for i, text in enumerate(sentence_texts):
        score = jaccard_similarity(text, opinion)
        if score > highest_score:
            highest_score = score
            best_match = text
            ret_id = i

    return ret_id, best_match


def preprocess_arabic(text):

    text = re.sub(r'[\u064B-\u0652]', '', text)

    text = text.replace('آ', 'ا').replace('أ', 'ا').replace('إ', 'ا')
    text = text.replace('ى', 'ي').replace('ؤ', 'و').replace('ئ', 'ي')
    text = text.replace('ة', 'ه')

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_most_similar_sentence_ARA(sentences, opinion):
    if opinion == "NAN":
        return -1, "NAN"

    sentence_texts = [s.partition(' ')[2] for s in sentences]

    sentence_texts_processed = [preprocess_arabic(s) for s in sentence_texts]
    opinion_processed = preprocess_arabic(opinion)

    best_match = ''
    highest_ratio = 0.0
    ret_id = 0

    for i, text in enumerate(sentence_texts_processed):
        ratio = difflib.SequenceMatcher(None, text, opinion_processed).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = sentence_texts[i]
            ret_id = i

    return ret_id, best_match


def clip(value):  # clip to a 0-1 value
    return max(0, min(1, value))


def get_system_provided_correct_na(prefix_file):
    file_path = prefix_file + '_detection_results.csv'
    df = pd.read_csv(file_path)

    filtered_df = df[(df["True Flag"] == 0) & (df["Predicted Flag"] == 0)]

    count = len(filtered_df)
    print("True Flag == 0 and Predicted Flag == 0 row number:", count)
    return count


def detect_with_multiagent(prefix_file, build_det_prompt, call_llms_api, sample_num):

    test_df = pd.read_csv("data/MEDEC-ALL-TestSet-with-GroundTruth-and-ErrorType.csv")
    grouped = test_df.head(sample_num).groupby("Text ID", sort=False)

    results = []

    # Task : Error Detection --> Error Flag

    for inx, (text_id, group) in enumerate(tqdm(grouped)):
        full_text = group["Text"].tolist()[0]
        prompts = build_det_prompt(full_text)
        true_flag = group["Error Flag"].iloc[0]

        agent_outputs = []

        for i in range(2):
            output = call_llms_api(prompts[i])
            agent_outputs.append(output)

        if agent_outputs[0]["result"] == agent_outputs[1]["result"] and agent_outputs[0]["result"] in ["CORRECT",
                                                                                                       "INCORRECT"]:
            final_result = agent_outputs[0]["result"]
            decision_source = "Agent1 + Agent2 Agreement"
            agent_outputs.append({"output": None, "result": None, "think": None, "confidence": None})
        else:
            agent3_output = call_llms_api(prompts[2])
            agent_outputs.append(agent3_output)
            final_result = agent3_output["result"]
            decision_source = "Agent3 Arbitration"

        pred_flag = 1 if final_result == "INCORRECT" else 0 if final_result == "CORRECT" else -1

        results.append({
            "Text ID": text_id,
            "True Flag": true_flag,
            "Predicted Flag": pred_flag,
            "Decision Source": decision_source,

            "Agent1 Output": agent_outputs[0]["output"],
            "Agent1 <think>": agent_outputs[0]["think"],
            "Agent1 Confidence": agent_outputs[0]["confidence"],

            "Agent2 Output": agent_outputs[1]["output"],
            "Agent2 <think>": agent_outputs[1]["think"],
            "Agent2 Confidence": agent_outputs[1]["confidence"],

            "Agent3 Output": agent_outputs[2]["output"],
            "Agent3 <think>": agent_outputs[2]["think"],
            "Agent3 Confidence": agent_outputs[2]["confidence"],
        })

        print('\n', results[-1])

    results_df = pd.DataFrame(results)
    results_df.to_csv(prefix_file + '_detection_results.csv', index=False)

    test_results = results_df[results_df["Predicted Flag"] != -1]
    accuracy = accuracy_score(test_results["True Flag"], test_results["Predicted Flag"])
    recall = recall_score(test_results["True Flag"], test_results["Predicted Flag"])
    print(f"\n✅ Error Detection Accuracy: {accuracy:.4f}")
    print(f"\n✅ Error Detection Recall: {recall:.4f}")


    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"]))
    print("\n📋 Classification Report:")
    print(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                target_names=["No Error", "Has Error"]))


    with open(prefix_file + "_all_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(f"\n✅ Error Detection Accuracy: {accuracy:.4f}\n")
        f.write(f"\n✅ Error Detection Recall: {recall:.4f}\n")

        f.write("\n📊 Confusion Matrix:\n")
        f.write(str(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"])))
        f.write("\n")

        f.write("\n📋 Classification Report:\n")
        f.write(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                      target_names=["No Error", "Has Error"]))


def detect_MA_debate(prefix_file, build_det_prompt, build_detection_debate_prompt, call_llms_api, sample_num,
                     input_file="data/Multi_Lang/MEDEC_merged_dataset_with_important_words.csv"):

    test_df = pd.read_csv(input_file)
    grouped = test_df.head(sample_num).groupby("Text ID", sort=False)

    results = []

    # Task : Error Detection --> Error Flag

    for inx, (text_id, group) in enumerate(tqdm(grouped)):
        full_text = group["Text"].tolist()[0]
        prompts = build_det_prompt(full_text)
        true_flag = group["Error Flag"].iloc[0]

        agent_outputs = []

        for i in range(2):
            output = call_llms_api(prompts[i])
            agent_outputs.append(output)

        if agent_outputs[0]["result"] == agent_outputs[1]["result"] and agent_outputs[0]["result"] in ["CORRECT",
                                                                                                       "INCORRECT"]:
            final_result = agent_outputs[0]["result"]
            decision_source = "Agent1 + Agent2 Agreement"
            agent_outputs.append({"output": None, "result": None, "think": None, "confidence": None})
        else:
            text = "\n\n".join(['Original Text:', full_text, 'Agent1 OUTPUT:', agent_outputs[0]['output'],
                                'Agent2 OUTPUT:', agent_outputs[1]['output']])
            prompts_debate = build_detection_debate_prompt(text)
            agent3_output = call_llms_api(prompts_debate[0])
            agent_outputs.append(agent3_output)
            final_result = agent3_output["result"]
            decision_source = "Agent3 Arbitration"

        pred_flag = 1 if final_result == "INCORRECT" else 0 if final_result == "CORRECT" else -1

        results.append({
            "Text ID": text_id,
            "True Flag": true_flag,
            "Predicted Flag": pred_flag,
            "Decision Source": decision_source,

            "Agent1 Output": agent_outputs[0]["output"],
            "Agent1 <think>": agent_outputs[0]["think"],
            "Agent1 Confidence": agent_outputs[0]["confidence"],

            "Agent2 Output": agent_outputs[1]["output"],
            "Agent2 <think>": agent_outputs[1]["think"],
            "Agent2 Confidence": agent_outputs[1]["confidence"],

            "Agent3 Output": agent_outputs[2]["output"],
            "Agent3 <think>": agent_outputs[2]["think"],
            "Agent3 Confidence": agent_outputs[2]["confidence"],
        })

        print('\n', results[-1])

    results_df = pd.DataFrame(results)
    results_df.to_csv(prefix_file + '_detection_results.csv', index=False)

    test_results = results_df[results_df["Predicted Flag"] != -1]
    accuracy = accuracy_score(test_results["True Flag"], test_results["Predicted Flag"])
    recall = recall_score(test_results["True Flag"], test_results["Predicted Flag"])
    print(f"\n✅ Error Detection Accuracy: {accuracy:.4f}")
    print(f"\n✅ Error Detection Recall: {recall:.4f}")

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"]))
    print("\n📋 Classification Report:")
    print(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                target_names=["No Error", "Has Error"]))

    with open(prefix_file + "_all_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(f"\n✅ Error Detection Accuracy: {accuracy:.4f}\n")
        f.write(f"\n✅ Error Detection Recall: {recall:.4f}\n")

        f.write("\n📊 Confusion Matrix:\n")
        f.write(str(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"])))
        f.write("\n")

        f.write("\n📋 Classification Report:\n")
        f.write(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                      target_names=["No Error", "Has Error"]))


def locate_and_correct_with_multiagent(prefix_file, build_location_promptA, call_llms_api,
                                       build_location_discuss_prompt, build_correction_prompt, sample_num,
                                       build_location_promptB=None):
    df = pd.read_csv(prefix_file + "_detection_results.csv")
    original_df = pd.read_csv("data/MEDEC-ALL-TestSet-with-GroundTruth-and-ErrorType.csv")

    error_df = df[df["Predicted Flag"] == 1].copy()
    # counters["system_provided_correct_na"] = counters["total_texts"] - len(error_df)

    error_df = error_df[:sample_num]

    results = []
    for _, row in tqdm(error_df.iterrows(), total=error_df.shape[0]):
        text_id = row["Text ID"]
        full_text = original_df[original_df["Text ID"] == text_id]["Text"].values[0]
        full_sentences = original_df[original_df["Text ID"] == text_id]["Sentences"].values[0].split('\r\n')
        gt_error_sentence = original_df[original_df["Text ID"] == text_id]["Error Sentence"].values[0]
        gt_error_sentence_id = original_df[original_df["Text ID"] == text_id]["Error Sentence ID"].values[0]
        gt_corrected_sentence = original_df[original_df["Text ID"] == text_id]["Corrected Sentence"].values[0]

        try:
            prompt_a = build_location_promptA(full_text, "Agent A")
            response_a = call_llms_api(prompt_a)
            id_a, opinion_a = find_most_similar_sentence(full_sentences, response_a)
        except Exception as e:
            print("API error from Agent A:", e)
            response_a = "ERROR"
            opinion_a = "ERROR"
            id_a = -2
        # time.sleep(10)

        try:
            if build_location_promptB == None:
                prompt_b = build_location_promptA(full_text, "Agent B")
            else:
                prompt_b = build_location_promptB(full_text, "Agent B")
            response_b = call_llms_api(prompt_b)
            id_b, opinion_b = find_most_similar_sentence(full_sentences, response_b)
        except Exception as e:
            print("API error from Agent B:", e)
            response_b = "ERROR"
            opinion_b = "ERROR"
            id_b = -2
        # time.sleep(10)

        final_response = "NAN"
        if id_a == id_b and opinion_a != "ERROR":
            predicted_error_sentence = opinion_a
            pred_id = id_a
        else:
            try:
                discussion_prompt_a = build_location_discuss_prompt("Agent C", full_text, 'Agent A Opinion:\n' + opinion_a + '\n\nAgent B Opinion:\n' + opinion_b)
                final_response = call_llms_api(discussion_prompt_a)
                pred_id, predicted_error_sentence = find_most_similar_sentence(full_sentences, final_response)
            except Exception as e:
                print("API error during agent discussion:", e)
                final_response = "ERROR"
                predicted_error_sentence = "ERROR"
                pred_id = -2
            # time.sleep(10)

        if predicted_error_sentence == "ERROR":
            continue

        try:
            if predicted_error_sentence != 'NAN':
                correction_prompt = build_correction_prompt(full_text, predicted_error_sentence)
                response2 = call_llms_api(correction_prompt)
                predicted_correction = response2
            else:
                predicted_correction = "NAN"
        except Exception as e:
            print("API error during correction:", e)
            predicted_correction = "ERROR"

        print("\n\nGroundTruth Error Sentence:", gt_error_sentence)
        print("Predicted Error Sentence:", predicted_error_sentence)
        print("GroundTruth Corrected Sentence:", gt_corrected_sentence)
        print("Predicted Corrected Sentence:", predicted_correction)

        results.append({
            "Text ID": text_id,
            "Predicted Error Sentence": predicted_error_sentence,
            "Predicted Error Sentence ID": pred_id,
            "GroundTruth Error Sentence": gt_error_sentence,
            "GroundTruth Error Sentence ID": gt_error_sentence_id,
            "Predicted Corrected Sentence": predicted_correction,
            "GroundTruth Corrected Sentence": gt_corrected_sentence,
            "Agent A Opinion": response_a,
            "Agent A ID": id_a,
            "Agent B Opinion": response_b,
            "Agent B ID": id_b,
            "Agent C Opinion": final_response,
            "Agent C ID": pred_id,
        })
        # time.sleep(10)

    result_df = pd.DataFrame(results)
    result_df.to_csv(prefix_file + "_correction_results.csv", index=False)
    print("\n✅ Saving multi-agent correction results:", prefix_file + "_correction_results.csv")


def locate_and_correct_with_multiagent_confidence(prefix_file, build_location_promptA, call_llms_api,
                                                  build_location_discuss_prompt, build_correction_prompt, sample_num,
                                                  build_location_promptB=None,
                                                  input_file="data/Multi_Lang/MEDEC_merged_dataset_with_important_words.csv"):
    df = pd.read_csv(prefix_file + "_detection_results.csv")
    original_df = pd.read_csv(input_file)

    # find_most_similar_sentence = find_most_similar_sentence_CN
    # find_most_similar_sentence = find_most_similar_sentence_ARA

    error_df = df[df["Predicted Flag"] == 1].copy()
    # counters["system_provided_correct_na"] = counters["total_texts"] - len(error_df)

    error_df = error_df[:sample_num]

    results = []
    for _, row in tqdm(error_df.iterrows(), total=error_df.shape[0]):
        text_id = row["Text ID"]
        full_text = original_df[original_df["Text ID"] == text_id]["Text"].values[0]
        full_sentences = original_df[original_df["Text ID"] == text_id]["Sentences"].values[0].split('\r\n')
        if len(full_sentences) <= 1:
            full_sentences = original_df[original_df["Text ID"] == text_id]["Sentences"].values[0].split('. ')
        gt_error_sentence = original_df[original_df["Text ID"] == text_id]["Error Sentence"].values[0]
        gt_error_sentence_id = original_df[original_df["Text ID"] == text_id]["Error Sentence ID"].values[0]
        gt_corrected_sentence = original_df[original_df["Text ID"] == text_id]["Corrected Sentence"].values[0]

        try:
            prompt_a = build_location_promptA(full_text, "Agent A")
            response_a = call_llms_api(prompt_a)
            id_a, opinion_a = find_most_similar_sentence(full_sentences, response_a['result'])
        except Exception as e:
            print("API error from Agent A:", e)
            response_a = "ERROR"
            opinion_a = "ERROR"
            id_a = -2
        # time.sleep(10)

        try:
            if build_location_promptB == None:
                prompt_b = build_location_promptA(full_text, "Agent B")
            else:
                prompt_b = build_location_promptB(full_text, "Agent B")
            response_b = call_llms_api(prompt_b)
            id_b, opinion_b = find_most_similar_sentence(full_sentences, response_b['result'])
        except Exception as e:
            print("API error from Agent B:", e)
            response_b = "ERROR"
            opinion_b = "ERROR"
            id_b = -2
        # time.sleep(10)

        final_response = "NAN"
        if id_a == id_b and opinion_a != "ERROR":
            predicted_error_sentence = opinion_a
            pred_id = id_a
        else:
            try:
                discussion_prompt_a = build_location_discuss_prompt("Agent C", full_text, 'Agent A Opinion:\n' + opinion_a + '\n\nAgent B Opinion:\n' + opinion_b)
                final_response = call_llms_api(discussion_prompt_a)
                pred_id, predicted_error_sentence = find_most_similar_sentence(full_sentences, final_response['result'])
            except Exception as e:
                print("API error during agent discussion:", e)
                final_response = "ERROR"
                predicted_error_sentence = "ERROR"
                pred_id = -2
            # time.sleep(10)

        if predicted_error_sentence == "ERROR":
            continue

        try:
            if predicted_error_sentence != 'NAN':
                correction_prompt = build_correction_prompt(full_text, predicted_error_sentence)
                response2 = call_llms_api(correction_prompt)
                predicted_correction = response2['result']
            else:
                predicted_correction = "NAN"
        except Exception as e:
            print("API error during correction:", e)
            predicted_correction = "ERROR"

        print("\n\nGroundTruth Error Sentence:", gt_error_sentence)
        print("Predicted Error Sentence:", predicted_error_sentence)
        print("GroundTruth Corrected Sentence:", gt_corrected_sentence)
        print("Predicted Corrected Sentence:", predicted_correction)

        results.append({
            "Text ID": text_id,
            "Predicted Error Sentence": predicted_error_sentence,
            "Predicted Error Sentence ID": pred_id,
            "GroundTruth Error Sentence": gt_error_sentence,
            "GroundTruth Error Sentence ID": gt_error_sentence_id,
            "Predicted Corrected Sentence": predicted_correction,
            "GroundTruth Corrected Sentence": gt_corrected_sentence,
            "Agent A Opinion": response_a,
            "Agent A ID": id_a,
            "Agent B Opinion": response_b,
            "Agent B ID": id_b,
            "Agent C Opinion": final_response,
            "Agent C ID": pred_id,
        })
        # time.sleep(10)

    result_df = pd.DataFrame(results)
    result_df.to_csv(prefix_file + "_correction_results.csv", index=False)
    print("\n✅ Saving multi-agent correction results:", prefix_file + "_correction_results.csv")


# hard matching
def evaluate_correction_with_sentences(prefix_file, counters, sample_num=None):
    df = pd.read_csv(prefix_file + "_correction_results.csv")
    if sample_num:
        df = df[:sample_num]

    location_acc = accuracy_score(
        df["GroundTruth Error Sentence"].fillna("NAN").str.strip(),
        df["Predicted Error Sentence"].fillna("NAN").str.strip()
    )
    print(f"\n🎯 Error Sentence Accuracy: {location_acc:.4f}")


    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    # rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleurt_model = bleurtscore.BleurtScorer(checkpoint="BLEURT-20")

    bleu_scores = []
    levenshtein_distances = []
    rouge_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rougeSU_scores = []
    bert_scores = []
    bleurt_scores = []

    refs = []
    hyps = []

    for _, row in df.iterrows():
        ref = row["GroundTruth Corrected Sentence"]
        hyp = row["Predicted Corrected Sentence"]
        if (ref == 'NAN' or pd.isna(ref)) and (hyp == 'NAN' or pd.isna(hyp)):
            counters["system_provided_correct_na"] += 1
        if (ref == 'NAN' or pd.isna(ref)) or (hyp == 'NAN' or pd.isna(hyp)):
            continue

        # BLEU
        ref_tokens = nltk.word_tokenize(ref)
        hyp_tokens = nltk.word_tokenize(hyp)
        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)

        # Levenshtein
        lev = Levenshtein.distance(ref, hyp)
        levenshtein_distances.append(lev)

        # ROUGE
        rouge_score = rouge.score(ref, hyp)
        rouge1_f1 = rouge_score['rouge1'].fmeasure
        rouge2_f1 = rouge_score['rouge2'].fmeasure
        rougeL_f1 = rouge_score['rougeL'].fmeasure
        # rougeSU4_f1 = rouge_score['rougeSU4'].fmeasure
        rouge_scores.append(rouge1_f1)
        rouge2_scores.append(rouge2_f1)
        rougeL_scores.append(rougeL_f1)
        # rougeSU_scores.append(rougeSU4_f1)

        # For BERTScore / BLEURT
        refs.append(ref)
        hyps.append(hyp)

    # BERTScore (batched)
    # P, R, F1 = bert_score.score(hyps, refs, lang="en", verbose=True)
    P, R, F1 = bert_score.score(hyps, refs, model_type='microsoft/deberta-xlarge-mnli',
                                lang='en', device='cpu', verbose=True,
                                rescale_with_baseline=True)   # roberta-large
    bert_scores = F1.numpy()
    ## clip scores to [0,1]
    bert_scores = np.array([clip(num) for num in bert_scores])

    # BLEURT (batched)
    bleurt_scores = bleurt_model.score(references=refs, candidates=hyps, batch_size=1)
    bleurtscores = np.array([clip(num) for num in bleurt_scores])
    composite_score_bleurt = (bleurtscores.sum() + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    composite_score_bleu = (sum(bleu_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_lev = sum(levenshtein_distances) / len(levenshtein_distances) if levenshtein_distances else 0.0

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    composite_score_rouge1 = (sum(rouge_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    composite_score_rouge2 = (sum(rouge2_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    composite_score_rougeL = (sum(rougeL_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    # avg_rougeSU = sum(rougeSU_scores) / len(rougeSU_scores) if rougeSU_scores else 0.0
    # composite_score_rougeSU = (sum(rougeSU_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_bert = sum(bert_scores) / len(bert_scores)
    composite_score_bert = (sum(bert_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_bleurt = bleurtscores.mean()

    print(f"\n📝 Ave BLEU score: {avg_bleu:.4f}")
    print(f"\n📝 Ave BLEUC score: {composite_score_bleu:.4f}")
    print(f"✂️ Ave Levenshtein dist: {avg_lev:.2f}")
    print(f"📘 Ave ROUGE1 score: {avg_rouge:.4f}")
    print(f"📘 R1FC score: {composite_score_rouge1:.4f}")
    print(f"📘 Ave ROUGE2 score: {avg_rouge2:.4f}")
    print(f"📘 R2FC score: {composite_score_rouge2:.4f}")
    print(f"📘 Ave rougeL score: {avg_rougeL:.4f}")
    print(f"📘 RLFC score: {composite_score_rougeL:.4f}")
    # print(f"📘 Ave rougeSU score: {avg_rougeSU:.4f}")
    # print(f"📘 RSUFC score: {composite_score_rougeSU:.4f}")
    print(f"🔍 Ave BERTScore F1: {avg_bert:.4f}")
    print(f"🔍 BERTC: {composite_score_bert:.4f}")
    print(f"📏 Ave BLEURT score: {avg_bleurt:.4f}")
    print(f"📏 BLEURTC score: {composite_score_bleurt:.4f}")

    with open(prefix_file + '_all_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n\n\n📝 Ave BLEU score: {avg_bleu:.4f}\n")
        file.write(f"📝 Ave BLEUC score: {composite_score_bleu:.4f}\n")
        file.write(f"✂️ Ave Levenshtein dist: {avg_lev:.2f}\n")
        file.write(f"📘 Ave ROUGE1 score: {avg_rouge:.4f}\n")
        file.write(f"📘 R1FC score: {composite_score_rouge1:.4f}\n")
        file.write(f"📘 Ave ROUGE2 score: {avg_rouge2:.4f}")
        file.write(f"📘 R2FC score: {composite_score_rouge2:.4f}")
        file.write(f"📘 Ave rougeL score: {avg_rougeL:.4f}")
        file.write(f"📘 RLFC score: {composite_score_rougeL:.4f}")
        # file.write(f"📘 Ave rougeSU score: {avg_rougeSU:.4f}")
        # file.write(f"📘 RSUFC score: {composite_score_rougeSU:.4f}")
        file.write(f"🔍 Ave BERTScore F1: {avg_bert:.4f}\n")
        file.write(f"🔍 BERTC: {composite_score_bert:.4f}\n")
        file.write(f"📏 Ave BLEURT score: {avg_bleurt:.4f}\n")
        file.write(f"📏 BLEURTC score: {composite_score_bleurt:.4f}\n")


# for CN baselines
def evaluate_correction_with_ids_CN(prefix_file, counters, sample_num=None):
    df = pd.read_csv(prefix_file + "_correction_results.csv")
    if sample_num:
        df = df[:sample_num]

    def normalize_value(val):
        if str(val).strip() == "-1":
            return -1
        return str(val).strip()

    count = sum(1 for a, b in zip(df["GroundTruth Error Sentence ID"], df["Predicted Error Sentence ID"]) if str(a) == b)
    location_acc = (count + counters["system_provided_correct_na"]) / counters["total_texts"]
    print(f"\n🎯  Error Sentence Accuracy: {location_acc:.4f}")

    with open(prefix_file + '_all_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n\n\n🎯  Error Sentence Accuracy: {location_acc:.4f}")

    for _, row in df.iterrows():
        ref = normalize_value(row["GroundTruth Error Sentence ID"])
        hyp = normalize_value(row["Predicted Error Sentence ID"])
        if ref == -1 and hyp == -1:
            counters["system_provided_correct_na"] += 1

    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    bleurt_model = bleurtscore.BleurtScorer(checkpoint="BLEURT-20")

    bleu_scores = []
    levenshtein_distances = []
    rouge_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bert_scores = []
    bleurt_scores = []
    metrics = NLGMetrics_CN()

    refs = []
    hyps = []

    for _, row in df.iterrows():
        ref = row["GroundTruth Corrected Sentence"]
        hyp = row["Predicted Corrected Sentence"]
        if (ref == 'NAN' or pd.isna(ref)) or (hyp == 'NAN' or pd.isna(hyp)):
            continue

        # BLEU
        bleu = metrics.compute_bleu([ref], [hyp])
        bleu_scores.append(bleu)

        # ROUGE
        rouge1_f1, rouge2_f1, rougeL_f1 = metrics.compute_rouge([ref], [hyp])
        rouge_scores.append(rouge1_f1)
        rouge2_scores.append(rouge2_f1)
        rougeL_scores.append(rougeL_f1)

        # For BERTScore / BLEURT
        refs.append(ref)
        hyps.append(hyp)

    # BERTScore (batched)
    F1 = metrics.compute_bertscore(refs, hyps)
    bert_scores = F1.numpy()
    bert_scores = np.array([clip(num) for num in bert_scores])

    # BLEURT (batched)
    bleurt_scores = metrics.compute_bleurt(refs, hyps)
    bleurtscores = np.array([clip(num) for num in bleurt_scores])
    composite_score_bleurt = (bleurtscores.sum() + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    composite_score_bleu = (sum(bleu_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    composite_score_rouge1 = (sum(rouge_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    composite_score_rouge2 = (sum(rouge2_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    composite_score_rougeL = (sum(rougeL_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_bert = sum(bert_scores) / len(bert_scores)
    composite_score_bert = (sum(bert_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_bleurt = bleurtscores.mean()

    print(f"\n📝 Ave BLEU score: {avg_bleu:.4f}")
    print(f"\n📝 Ave BLEUC score: {composite_score_bleu:.4f}")
    print(f"📘 Ave ROUGE1 score: {avg_rouge:.4f}")
    print(f"📘 R1FC score: {composite_score_rouge1:.4f}")
    print(f"📘 Ave ROUGE2 score: {avg_rouge2:.4f}")
    print(f"📘 R2FC score: {composite_score_rouge2:.4f}")
    print(f"📘 Ave rougeL score: {avg_rougeL:.4f}")
    print(f"📘 RLFC score: {composite_score_rougeL:.4f}")
    print(f"🔍 Ave BERTScore F1: {avg_bert:.4f}")
    print(f"🔍 BERTC: {composite_score_bert:.4f}")
    print(f"📏 Ave BLEURT score: {avg_bleurt:.4f}")
    print(f"📏 BLEURTC score: {composite_score_bleurt:.4f}")

    with open(prefix_file + '_all_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n\n\n📝 Ave BLEU score: {avg_bleu:.4f}\n")
        file.write(f"📝 Ave BLEUC score: {composite_score_bleu:.4f}\n")
        file.write(f"📘 Ave ROUGE1 score: {avg_rouge:.4f}\n")
        file.write(f"📘 R1FC score: {composite_score_rouge1:.4f}\n")
        file.write(f"📘 Ave ROUGE2 score: {avg_rouge2:.4f}")
        file.write(f"📘 R2FC score: {composite_score_rouge2:.4f}")
        file.write(f"📘 Ave rougeL score: {avg_rougeL:.4f}")
        file.write(f"📘 RLFC score: {composite_score_rougeL:.4f}")
        file.write(f"🔍 Ave BERTScore F1: {avg_bert:.4f}\n")
        file.write(f"🔍 BERTC: {composite_score_bert:.4f}\n")
        file.write(f"📏 Ave BLEURT score: {avg_bleurt:.4f}\n")
        file.write(f"📏 BLEURTC score: {composite_score_bleurt:.4f}\n")


def evaluate_correction_with_ids_Arabic(prefix_file, counters, sample_num=None):
    df = pd.read_csv(prefix_file + "_correction_results.csv")
    if sample_num:
        df = df[:sample_num]

    count = sum(1 for a, b in zip(df["GroundTruth Error Sentence ID"], df["Predicted Error Sentence ID"]) if a == b)
    location_acc = (count + counters["system_provided_correct_na"]) / counters["total_texts"]
    print(f"\n🎯  Error Sentence Accuracy: {location_acc:.4f}")

    with open(prefix_file + '_all_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n\n\n🎯  Error Sentence Accuracy: {location_acc:.4f}")

    for _, row in df.iterrows():
        ref = row["GroundTruth Error Sentence ID"]
        hyp = row["Predicted Error Sentence ID"]
        if ref == -1 and hyp == -1:
            counters["system_provided_correct_na"] += 1

    smoothie = SmoothingFunction().method4
    rouge = Rouge()
    bleurt_model = bleurtscore.BleurtScorer(checkpoint="BLEURT-20")

    bleu_scores = []
    levenshtein_distances = []
    rouge_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rougeSU_scores = []
    bert_scores = []
    bleurt_scores = []
    metrics = NLGMetrics_CN()

    refs = []
    hyps = []

    for _, row in df.iterrows():
        ref = row["GroundTruth Corrected Sentence"]
        hyp = row["Predicted Corrected Sentence"]
        if (ref == 'NAN' or pd.isna(ref)) or (hyp == 'NAN' or pd.isna(hyp)):
            continue

        # BLEU
        ref_tokens = nltk.word_tokenize(ref)
        hyp_tokens = nltk.word_tokenize(hyp)
        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)

        # ROUGE
        rouge_score = rouge.get_scores(ref, hyp)[0]
        rouge1_f1 = rouge_score['rouge-1']['f']
        rouge2_f1 = rouge_score['rouge-2']['f']
        rougeL_f1 = rouge_score['rouge-l']['f']
        rouge_scores.append(rouge1_f1)
        rouge2_scores.append(rouge2_f1)
        rougeL_scores.append(rougeL_f1)

        # For BERTScore / BLEURT
        refs.append(ref)
        hyps.append(hyp)

    # BERTScore (batched)
    # P, R, F1 = bert_score.score(hyps, refs, lang="ar", verbose=True)
    P, R, F1 = bert_score.score(hyps, refs, model_type='microsoft/deberta-xlarge-mnli',
                                lang='ar', device='cpu', verbose=True,
                                rescale_with_baseline=True)  # roberta-large
    bert_scores = F1.numpy()
    ## clip scores to [0,1]
    bert_scores = np.array([clip(num) for num in bert_scores])

    # BLEURT (batched)
    bleurt_scores = bleurt_model.score(references=refs, candidates=hyps, batch_size=1)
    bleurtscores = np.array([clip(num) for num in bleurt_scores])
    composite_score_bleurt = (bleurtscores.sum() + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    composite_score_bleu = (sum(bleu_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    composite_score_rouge1 = (sum(rouge_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    composite_score_rouge2 = (sum(rouge2_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    composite_score_rougeL = (sum(rougeL_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_bert = sum(bert_scores) / len(bert_scores)
    composite_score_bert = (sum(bert_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_bleurt = bleurtscores.mean()

    print(f"\n📝 Ave BLEU score: {avg_bleu:.4f}")
    print(f"\n📝 Ave BLEUC score: {composite_score_bleu:.4f}")
    print(f"📘 Ave ROUGE1 score: {avg_rouge:.4f}")
    print(f"📘 R1FC score: {composite_score_rouge1:.4f}")
    print(f"📘 Ave ROUGE2 score: {avg_rouge2:.4f}")
    print(f"📘 R2FC score: {composite_score_rouge2:.4f}")
    print(f"📘 Ave rougeL score: {avg_rougeL:.4f}")
    print(f"📘 RLFC score: {composite_score_rougeL:.4f}")
    print(f"🔍 Ave BERTScore F1: {avg_bert:.4f}")
    print(f"🔍 BERTC: {composite_score_bert:.4f}")
    print(f"📏 Ave BLEURT score: {avg_bleurt:.4f}")
    print(f"📏 BLEURTC score: {composite_score_bleurt:.4f}")

    with open(prefix_file + '_all_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n\n\n📝 Ave BLEU score: {avg_bleu:.4f}\n")
        file.write(f"📝 Ave BLEUC score: {composite_score_bleu:.4f}\n")
        file.write(f"📘 Ave ROUGE1 score: {avg_rouge:.4f}\n")
        file.write(f"📘 R1FC score: {composite_score_rouge1:.4f}\n")
        file.write(f"📘 Ave ROUGE2 score: {avg_rouge2:.4f}")
        file.write(f"📘 R2FC score: {composite_score_rouge2:.4f}")
        file.write(f"📘 Ave rougeL score: {avg_rougeL:.4f}")
        file.write(f"📘 RLFC score: {composite_score_rougeL:.4f}")
        file.write(f"🔍 Ave BERTScore F1: {avg_bert:.4f}\n")
        file.write(f"🔍 BERTC: {composite_score_bert:.4f}\n")
        file.write(f"📏 Ave BLEURT score: {avg_bleurt:.4f}\n")
        file.write(f"📏 BLEURTC score: {composite_score_bleurt:.4f}\n")


def evaluate_correction_with_ids_EN(prefix_file, counters, sample_num=None):
    df = pd.read_csv(prefix_file + "_correction_results.csv")
    if sample_num:
        df = df[:sample_num]

    def normalize_value(val):
        if str(val).strip() == "-1":
            return -1
        return str(val).strip()

    df["Predicted Error Sentence ID"] = pd.to_numeric(
        df["Predicted Error Sentence ID"], errors="coerce"
    ).fillna(-2).astype(int)
    count = sum(1 for a, b in zip(df["GroundTruth Error Sentence ID"], df["Predicted Error Sentence ID"]) if a == b)
    location_acc = (count + counters["system_provided_correct_na"]) / counters["total_texts"]
    print(f"\n🎯  Error Sentence Accuracy: {location_acc:.4f}")

    with open(prefix_file + '_all_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n\n\n🎯  Error Sentence Accuracy: {location_acc:.4f}")

    for _, row in df.iterrows():
        ref = normalize_value(row["GroundTruth Error Sentence ID"])
        hyp = normalize_value(row["Predicted Error Sentence ID"])
        if ref == -1 and hyp == -1:
            counters["system_provided_correct_na"] += 1

    smoothie = SmoothingFunction().method4
    rouge = Rouge()
    bleurt_model = bleurtscore.BleurtScorer(checkpoint="BLEURT-20")

    bleu_scores = []
    levenshtein_distances = []
    rouge_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rougeSU_scores = []
    bert_scores = []
    bleurt_scores = []
    metrics = NLGMetrics_CN()

    refs = []
    hyps = []

    for _, row in df.iterrows():
        ref = row["GroundTruth Corrected Sentence"]
        hyp = row["Predicted Corrected Sentence"]
        if (ref == 'NAN' or pd.isna(ref)) or (hyp == 'NAN' or pd.isna(hyp)):
            continue

        # BLEU
        ref_tokens = nltk.word_tokenize(ref)
        hyp_tokens = nltk.word_tokenize(hyp)
        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)

        # ROUGE
        rouge_score = rouge.get_scores(ref, hyp)[0]
        rouge1_f1 = rouge_score['rouge-1']['f']
        rouge2_f1 = rouge_score['rouge-2']['f']
        rougeL_f1 = rouge_score['rouge-l']['f']
        rouge_scores.append(rouge1_f1)
        rouge2_scores.append(rouge2_f1)
        rougeL_scores.append(rougeL_f1)

        # For BERTScore / BLEURT
        refs.append(ref)
        hyps.append(hyp)

    # BERTScore (batched)
    # P, R, F1 = bert_score.score(hyps, refs, lang="en", verbose=True)
    P, R, F1 = bert_score.score(hyps, refs, model_type='microsoft/deberta-xlarge-mnli',
                                lang='en', device='cpu', verbose=True,
                                rescale_with_baseline=True)  # roberta-large
    bert_scores = F1.numpy()
    ## clip scores to [0,1]
    bert_scores = np.array([clip(num) for num in bert_scores])

    # BLEURT (batched)
    bleurt_scores = bleurt_model.score(references=refs, candidates=hyps, batch_size=1)
    bleurtscores = np.array([clip(num) for num in bleurt_scores])
    composite_score_bleurt = (bleurtscores.sum() + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    composite_score_bleu = (sum(bleu_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    composite_score_rouge1 = (sum(rouge_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    composite_score_rouge2 = (sum(rouge2_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    composite_score_rougeL = (sum(rougeL_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_bert = sum(bert_scores) / len(bert_scores)
    composite_score_bert = (sum(bert_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_bleurt = bleurtscores.mean()

    print(f"\n📝 Ave BLEU score: {avg_bleu:.4f}")
    print(f"\n📝 Ave BLEUC score: {composite_score_bleu:.4f}")
    print(f"📘 Ave ROUGE1 score: {avg_rouge:.4f}")
    print(f"📘 R1FC score: {composite_score_rouge1:.4f}")
    print(f"📘 Ave ROUGE2 score: {avg_rouge2:.4f}")
    print(f"📘 R2FC score: {composite_score_rouge2:.4f}")
    print(f"📘 Ave rougeL score: {avg_rougeL:.4f}")
    print(f"📘 RLFC score: {composite_score_rougeL:.4f}")
    print(f"🔍 Ave BERTScore F1: {avg_bert:.4f}")
    print(f"🔍 BERTC: {composite_score_bert:.4f}")
    print(f"📏 Ave BLEURT score: {avg_bleurt:.4f}")
    print(f"📏 BLEURTC score: {composite_score_bleurt:.4f}")

    with open(prefix_file + '_all_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n\n\n📝 Ave BLEU score: {avg_bleu:.4f}\n")
        file.write(f"📝 Ave BLEUC score: {composite_score_bleu:.4f}\n")
        file.write(f"📘 Ave ROUGE1 score: {avg_rouge:.4f}\n")
        file.write(f"📘 R1FC score: {composite_score_rouge1:.4f}\n")
        file.write(f"📘 Ave ROUGE2 score: {avg_rouge2:.4f}")
        file.write(f"📘 R2FC score: {composite_score_rouge2:.4f}")
        file.write(f"📘 Ave rougeL score: {avg_rougeL:.4f}")
        file.write(f"📘 RLFC score: {composite_score_rougeL:.4f}")
        file.write(f"🔍 Ave BERTScore F1: {avg_bert:.4f}\n")
        file.write(f"🔍 BERTC: {composite_score_bert:.4f}\n")
        file.write(f"📏 Ave BLEURT score: {avg_bleurt:.4f}\n")
        file.write(f"📏 BLEURTC score: {composite_score_bleurt:.4f}\n")


# soft matching
def evaluate_correction_with_ids(prefix_file, counters, sample_num=None):
    df = pd.read_csv(prefix_file + "_correction_results.csv")
    if sample_num:
        df = df[:sample_num]

    count = sum(1 for a, b in zip(df["GroundTruth Error Sentence ID"], df["Predicted Error Sentence ID"]) if a == b)
    location_acc = (count + counters["system_provided_correct_na"]) / counters["total_texts"]
    print(f"\n🎯  Error Sentence Accuracy): {location_acc:.4f}")

    with open(prefix_file + '_all_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n\n\n🎯  Error Sentence Accuracy): {location_acc:.4f}")

    for _, row in df.iterrows():
        ref = row["GroundTruth Error Sentence ID"]
        hyp = row["Predicted Error Sentence ID"]
        if ref == -1 and hyp == -1:
            counters["system_provided_correct_na"] += 1

    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleurt_model = bleurtscore.BleurtScorer(checkpoint="BLEURT-20")

    bleu_scores = []
    levenshtein_distances = []
    rouge_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rougeSU_scores = []
    bert_scores = []
    bleurt_scores = []

    refs = []
    hyps = []

    for _, row in df.iterrows():
        ref = row["GroundTruth Corrected Sentence"]
        hyp = row["Predicted Corrected Sentence"]
        if (ref == 'NAN' or pd.isna(ref)) or (hyp == 'NAN' or pd.isna(hyp)):
            continue

        # BLEU
        ref_tokens = nltk.word_tokenize(ref)
        hyp_tokens = nltk.word_tokenize(hyp)
        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu)

        # Levenshtein
        lev = Levenshtein.distance(ref, hyp)
        levenshtein_distances.append(lev)

        # ROUGE
        rouge_score = rouge.score(ref, hyp)
        rouge1_f1 = rouge_score['rouge1'].fmeasure
        rouge2_f1 = rouge_score['rouge2'].fmeasure
        rougeL_f1 = rouge_score['rougeL'].fmeasure
        # rougeSU4_f1 = rouge_score['rougeSU4'].fmeasure
        rouge_scores.append(rouge1_f1)
        rouge2_scores.append(rouge2_f1)
        rougeL_scores.append(rougeL_f1)
        # rougeSU_scores.append(rougeSU4_f1)

        # For BERTScore / BLEURT
        refs.append(ref)
        hyps.append(hyp)

    # BERTScore (batched)
    # P, R, F1 = bert_score.score(hyps, refs, lang="en", verbose=True)
    P, R, F1 = bert_score.score(hyps, refs, model_type='microsoft/deberta-xlarge-mnli',
                                lang='en', device='cpu', verbose=True,
                                rescale_with_baseline=True)  # roberta-large
    bert_scores = F1.numpy()
    ## clip scores to [0,1]
    bert_scores = np.array([clip(num) for num in bert_scores])

    # BLEURT (batched)
    bleurt_scores = bleurt_model.score(references=refs, candidates=hyps, batch_size=1)
    bleurtscores = np.array([clip(num) for num in bleurt_scores])
    composite_score_bleurt = (bleurtscores.sum() + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    composite_score_bleu = (sum(bleu_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_lev = sum(levenshtein_distances) / len(levenshtein_distances) if levenshtein_distances else 0.0

    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    composite_score_rouge1 = (sum(rouge_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    composite_score_rouge2 = (sum(rouge2_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    composite_score_rougeL = (sum(rougeL_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    # avg_rougeSU = sum(rougeSU_scores) / len(rougeSU_scores) if rougeSU_scores else 0.0
    # composite_score_rougeSU = (sum(rougeSU_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]

    avg_bert = sum(bert_scores) / len(bert_scores)
    composite_score_bert = (sum(bert_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]
    avg_bleurt = bleurtscores.mean()

    print(f"\n📝 Ave BLEU score: {avg_bleu:.4f}")
    print(f"\n📝 Ave BLEUC score: {composite_score_bleu:.4f}")
    print(f"✂️ Ave Levenshtein dist: {avg_lev:.2f}")
    print(f"📘 Ave ROUGE1 score: {avg_rouge:.4f}")
    print(f"📘 R1FC score: {composite_score_rouge1:.4f}")
    print(f"📘 Ave ROUGE2 score: {avg_rouge2:.4f}")
    print(f"📘 R2FC score: {composite_score_rouge2:.4f}")
    print(f"📘 Ave rougeL score: {avg_rougeL:.4f}")
    print(f"📘 RLFC score: {composite_score_rougeL:.4f}")
    # print(f"📘 Ave rougeSU score: {avg_rougeSU:.4f}")
    # print(f"📘 RSUFC score: {composite_score_rougeSU:.4f}")
    print(f"🔍 Ave BERTScore F1: {avg_bert:.4f}")
    print(f"🔍 BERTC: {composite_score_bert:.4f}")
    print(f"📏 Ave BLEURT score: {avg_bleurt:.4f}")
    print(f"📏 BLEURTC score: {composite_score_bleurt:.4f}")

    with open(prefix_file + '_all_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"\n\n\n📝 Ave BLEU score: {avg_bleu:.4f}\n")
        file.write(f"📝 Ave BLEUC score: {composite_score_bleu:.4f}\n")
        file.write(f"✂️ Ave Levenshtein dist: {avg_lev:.2f}\n")
        file.write(f"📘 Ave ROUGE1 score: {avg_rouge:.4f}\n")
        file.write(f"📘 R1FC score: {composite_score_rouge1:.4f}\n")
        file.write(f"📘 Ave ROUGE2 score: {avg_rouge2:.4f}")
        file.write(f"📘 R2FC score: {composite_score_rouge2:.4f}")
        file.write(f"📘 Ave rougeL score: {avg_rougeL:.4f}")
        file.write(f"📘 RLFC score: {composite_score_rougeL:.4f}")
        # file.write(f"📘 Ave rougeSU score: {avg_rougeSU:.4f}")
        # file.write(f"📘 RSUFC score: {composite_score_rougeSU:.4f}")
        file.write(f"🔍 Ave BERTScore F1: {avg_bert:.4f}\n")
        file.write(f"🔍 BERTC: {composite_score_bert:.4f}\n")
        file.write(f"📏 Ave BLEURT score: {avg_bleurt:.4f}\n")
        file.write(f"📏 BLEURTC score: {composite_score_bleurt:.4f}\n")


def evaluate_correction_keywords(prefix_file, counters, sample_num=None,
                                 input_file="data/Multi_Lang/MEDEC_merged_dataset_with_important_words.csv"):

    df = pd.read_csv(prefix_file + "_correction_results.csv")
    original_df = pd.read_csv(input_file)

    if sample_num:
        df = df[:sample_num]
        original_df = original_df[:sample_num]

    for _, row in df.iterrows():
        ref = row["GroundTruth Error Sentence ID"]
        hyp = row["Predicted Error Sentence ID"]
        if ref == -1 and hyp == -1:
            counters["system_provided_correct_na"] += 1

    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleurt_model = bleurtscore.BleurtScorer(checkpoint="BLEURT-20")

    rouge_scores = []
    refs = []
    hyps = []

    keywords_exist = []
    alpha = 0.5

    for i, row in df.iterrows():
        # print('Processing', i, 'case:', row['Text ID'])
        ref = row["GroundTruth Corrected Sentence"]
        hyp = row["Predicted Corrected Sentence"]
        if (ref == 'NAN' or pd.isna(ref)) or (hyp == 'NAN' or pd.isna(hyp)):
            continue

        keyword = original_df[original_df['Text ID'] == row['Text ID']]['important words'].iloc[0]
        if keyword in hyp:
            keywords_exist.append(1)
        else:
            keywords_exist.append(0)

        # ROUGE
        rouge_score = rouge.score(ref, hyp)
        rouge1_f1 = rouge_score['rouge1'].fmeasure
        rouge_scores.append(rouge1_f1)

        # For BERTScore / BLEURT
        refs.append(ref)
        hyps.append(hyp)

    # BERTScore (batched)
    # P, R, F1 = bert_score.score(hyps, refs, lang="en", verbose=True)
    P, R, F1 = bert_score.score(hyps, refs, model_type='microsoft/deberta-xlarge-mnli',
                                lang='en', device='cpu', verbose=True,
                                rescale_with_baseline=True)  # roberta-large
    bert_scores = F1.numpy()
    ## clip scores to [0,1]
    bert_scores = np.array([clip(num) for num in bert_scores])

    # BLEURT (batched)
    bleurt_scores = bleurt_model.score(references=refs, candidates=hyps, batch_size=1)
    bleurtscores = np.array([clip(num) for num in bleurt_scores])

    # Core: The calculation of keywords emphasised scores
    keywords_scores = keywords_exist * (alpha + (rouge_scores + bert_scores + bleurtscores)/3 * (1-alpha))
    avg_keywords_scores = sum(keywords_scores) / len(keywords_scores)
    composite_keywords_scores = (sum(keywords_scores) + counters["system_provided_correct_na"]) / counters["total_texts"]

    print(f"\n📝 Ave keywords_scores score: {avg_keywords_scores:.4f}")
    print(f"\n📝 composite_keywords_scores score: {composite_keywords_scores:.4f}")

    with open(prefix_file + '_all_evaluation.txt', 'a', encoding='utf-8') as file:
        file.write(f"📝 Ave keywords_scores score: {avg_keywords_scores:.4f}\n")
        file.write(f"📝 composite_keywords_scores score: {composite_keywords_scores:.4f}\n")


def baseline_detection_correction(prefix_file, build_prompt, call_llms_api, parse_id_correction, sample_num,
                                  input_file="data/Multi_Lang/MEDEC_merged_dataset_with_important_words.csv"):
    test_df = pd.read_csv(input_file)
    grouped = test_df.head(sample_num).groupby("Text ID", sort=False)
    results = []
    results_correct = []

    # Task : Error Detection --> Error Flag
    for inx, (text_id, group) in enumerate(tqdm(grouped)):
        full_text = group["Text"].tolist()[0]
        full_sentences = group["Sentences"].values[0].split('\r\n')
        gt_error_sentence = group["Error Sentence"].values[0]
        gt_error_sentence_id = group["Error Sentence ID"].values[0]
        gt_corrected_sentence = group["Corrected Sentence"].values[0]
        true_flag = group["Error Flag"].iloc[0]

        # prompts = build_prompt(full_text)
        prompts = build_prompt(text_id, full_text)

        prompt = prompts[0]

        tolerance = 2
        while tolerance > 0:
            prediction = call_llms_api(prompt)

            if "CORRECT" in prediction:
                pred_flag = 0
                results_correct.append({
                    "Text ID": text_id,
                    "Predicted Error Sentence": 'NAN',
                    "GroundTruth Error Sentence": gt_error_sentence,
                    "Predicted Corrected Sentence": 'NAN',
                    "GroundTruth Corrected Sentence": gt_corrected_sentence,
                })
                break
            else:
                pred_flag = 1
                try:
                    error_sent, correct_sent = parse_id_correction(prediction)
                except Exception as e:
                    tolerance -= 1
                    print("\nAgent error:", e, "  Tolerance:", tolerance)
                    continue

                results_correct.append({
                    "Text ID": text_id,
                    "Predicted Error Sentence": error_sent,
                    "GroundTruth Error Sentence": gt_error_sentence,
                    "Predicted Corrected Sentence": correct_sent,
                    "GroundTruth Corrected Sentence": gt_corrected_sentence,
                })
                break

        if tolerance <= 0:
            pred_flag = 0
            results_correct.append({
                "Text ID": text_id,
                "Predicted Error Sentence": 'NAN',
                "GroundTruth Error Sentence": gt_error_sentence,
                "Predicted Corrected Sentence": 'NAN',
                "GroundTruth Corrected Sentence": gt_corrected_sentence,
            })

        results.append(
            {"Text ID": text_id, "True Flag": true_flag, "Predicted Flag": pred_flag, "Output": prediction})
        print('\n', results[-1])

    results_correct_df = pd.DataFrame(results_correct)
    results_correct_df.to_csv(prefix_file + "_correction_results.csv", index=False)
    print("\n✅Saving multi-agent correction results:", prefix_file + "_correction_results.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(prefix_file + "_detection_results.csv", index=False)

    test_results = results_df[results_df["Predicted Flag"] != -1]
    accuracy = accuracy_score(test_results["True Flag"], test_results["Predicted Flag"])
    recall = recall_score(test_results["True Flag"], test_results["Predicted Flag"])
    print(f"\n✅ Error Detection Accuracy: {accuracy:.4f}")
    print(f"\n✅ Error Detection Recall: {recall:.4f}")

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"]))
    print("\n📋 Classification Report:")
    print(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                target_names=["No Error", "Has Error"]))

    with open(prefix_file + "_all_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(f"\n✅ Error Detection Accuracy: {accuracy:.4f}\n")
        f.write(f"\n✅ Error Detection Recall: {recall:.4f}\n")

        f.write("\n📊 Confusion Matrix:\n")
        f.write(str(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"])))
        f.write("\n")

        f.write("\n📋 Classification Report:\n")
        f.write(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                      target_names=["No Error", "Has Error"]))


def baseline_detection_correction_confidence(prefix_file, build_prompt, call_llms_api, parse_id_correction, sample_num):
    # test_df = pd.read_csv("data/MEDEC-ALL-TestSet-with-GroundTruth-and-ErrorType.csv")
    test_df = pd.read_csv("data/reviewed_data_ARA_test.csv")
    grouped = test_df.head(sample_num).groupby("Text ID", sort=False)
    results = []
    results_correct = []

    # Task : Error Detection --> Error Flag
    for inx, (text_id, group) in enumerate(tqdm(grouped)):
        full_text = group["Text"].tolist()[0]
        full_sentences = group["Sentences"].values[0].split('\r\n')
        gt_error_sentence = group["Error Sentence"].values[0]
        gt_error_sentence_id = group["Error Sentence ID"].values[0]
        gt_corrected_sentence = group["Corrected Sentence"].values[0]
        true_flag = group["Error Flag"].iloc[0]

        prompts = build_prompt(full_text)

        prompt = prompts[0]

        tolerance = 2
        while tolerance > 0:
            prediction = call_llms_api(prompt)

            if "CORRECT" in prediction:
                pred_flag = 0
                results_correct.append({
                    "Text ID": text_id,
                    "Predicted Error Sentence": 'NAN',
                    "GroundTruth Error Sentence": gt_error_sentence,
                    "Predicted Corrected Sentence": 'NAN',
                    "GroundTruth Corrected Sentence": gt_corrected_sentence,
                    "flag_confidence": -1,
                    "location_confidence": -1,
                    "correction_confidence": -1,
                })
                break
            else:
                pred_flag = 1
                try:
                    error_sent, flag_confidence, location_confidence, correction_confidence, correct_sent =\
                        parse_id_correction(prediction)
                except Exception as e:
                    tolerance -= 1
                    print("\nAgent error:", e, "  Tolerance:", tolerance)
                    continue

                results_correct.append({
                    "Text ID": text_id,
                    "Predicted Error Sentence": error_sent,
                    "GroundTruth Error Sentence": gt_error_sentence,
                    "Predicted Corrected Sentence": correct_sent,
                    "GroundTruth Corrected Sentence": gt_corrected_sentence,
                    "flag_confidence": flag_confidence,
                    "location_confidence": location_confidence,
                    "correction_confidence": correction_confidence,
                })
                break

        if tolerance <= 0:
            pred_flag = 0
            results_correct.append({
                "Text ID": text_id,
                "Predicted Error Sentence": 'NAN',
                "GroundTruth Error Sentence": gt_error_sentence,
                "Predicted Corrected Sentence": 'NAN',
                "GroundTruth Corrected Sentence": gt_corrected_sentence,
            })

        results.append(
            {"Text ID": text_id, "True Flag": true_flag, "Predicted Flag": pred_flag, "Output": prediction})
        print('\n', results[-1])

    results_correct_df = pd.DataFrame(results_correct)
    results_correct_df.to_csv(prefix_file + "_correction_results.csv", index=False)
    print("\n✅Saving multi-agent correction results:", prefix_file + "_correction_results.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(prefix_file + "_detection_results.csv", index=False)

    test_results = results_df[results_df["Predicted Flag"] != -1]
    accuracy = accuracy_score(test_results["True Flag"], test_results["Predicted Flag"])
    recall = recall_score(test_results["True Flag"], test_results["Predicted Flag"])
    print(f"\n✅ Error Detection Accuracy: {accuracy:.4f}")
    print(f"\n✅ Error Detection Recall: {recall:.4f}")

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"]))
    print("\n📋 Classification Report:")
    print(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                target_names=["No Error", "Has Error"]))

    with open(prefix_file + "_all_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(f"\n✅ Error Detection Accuracy: {accuracy:.4f}\n")
        f.write(f"\n✅ Error Detection Recall: {recall:.4f}\n")

        f.write("\n📊 Confusion Matrix:\n")
        f.write(str(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"])))
        f.write("\n")

        f.write("\n📋 Classification Report:\n")
        f.write(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                      target_names=["No Error", "Has Error"]))


def baseline_detection_correction_CN(prefix_file, build_prompt, call_llms_api, parse_id_correction, sample_num,
                                     input_file="data/reviewed_dataset_EN_test.csv"):

    test_df = pd.read_csv(input_file)
    grouped = test_df.head(sample_num).groupby("Text ID", sort=False)
    results = []
    results_correct = []

    # Task : Error Detection --> Error Flag
    for inx, (text_id, group) in enumerate(tqdm(grouped)):
        full_text = group["Text"].tolist()[0]
        full_sentences = group["Sentences"].values[0].split('\r\n')
        gt_error_sentence = group["Error Sentence"].values[0]
        gt_error_sentence_id = group["Error Sentence ID"].values[0]
        gt_corrected_sentence = group["Corrected Sentence"].values[0]
        true_flag = group["Error Flag"].iloc[0]

        prompts = build_prompt(text_id, full_text)

        prompt = prompts[0]

        tolerance = 2
        while tolerance > 0:
            prediction = call_llms_api(prompt)

            if "NAN" in prediction:
                pred_flag = 0
                results_correct.append({
                    "Text ID": text_id,
                    "Predicted Error Sentence": 'NAN',
                    "Predicted Error Sentence ID": -1,
                    "GroundTruth Error Sentence": gt_error_sentence,
                    "GroundTruth Error Sentence ID": gt_error_sentence_id,
                    "Predicted Corrected Sentence": 'NAN',
                    "GroundTruth Corrected Sentence": gt_corrected_sentence,
                })
                break
            else:
                pred_flag = 1
                try:
                    pred_id, error_sent, correct_sent = parse_id_correction(prediction)
                except Exception as e:
                    tolerance -= 1
                    print("\nAgent error:", e, "  Tolerance:", tolerance)
                    continue

                results_correct.append({
                    "Text ID": text_id,
                    "Predicted Error Sentence": pred_id,
                    "Predicted Error Sentence ID": pred_id,
                    "GroundTruth Error Sentence": gt_error_sentence,
                    "GroundTruth Error Sentence ID": gt_error_sentence_id,
                    "Predicted Corrected Sentence": correct_sent,
                    "GroundTruth Corrected Sentence": gt_corrected_sentence,
                })
                break

        if tolerance <= 0:
            pred_flag = 0
            results_correct.append({
                "Text ID": text_id,
                "Predicted Error Sentence": 'NAN',
                "Predicted Error Sentence ID": -1,
                "GroundTruth Error Sentence": gt_error_sentence,
                "GroundTruth Error Sentence ID": gt_error_sentence_id,
                "Predicted Corrected Sentence": 'NAN',
                "GroundTruth Corrected Sentence": gt_corrected_sentence,
            })

        results.append(
            {"Text ID": text_id, "True Flag": true_flag, "Predicted Flag": pred_flag, "Output": prediction})
        print('\n', results[-1])

    results_correct_df = pd.DataFrame(results_correct)
    results_correct_df.to_csv(prefix_file + "_correction_results.csv", index=False)
    print("\n✅Saving multi-agent correction results:", prefix_file + "_correction_results.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(prefix_file + "_detection_results.csv", index=False)

    test_results = results_df[results_df["Predicted Flag"] != -1]
    accuracy = accuracy_score(test_results["True Flag"], test_results["Predicted Flag"])
    recall = recall_score(test_results["True Flag"], test_results["Predicted Flag"])
    print(f"\n✅ Error Detection Accuracy: {accuracy:.4f}")
    print(f"\n✅ Error Detection Recall: {recall:.4f}")

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"]))
    print("\n📋 Classification Report:")
    print(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                target_names=["No Error", "Has Error"]))

    with open(prefix_file + "_all_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(f"\n✅ Error Detection Accuracy: {accuracy:.4f}\n")
        f.write(f"\n✅ Error Detection Recall: {recall:.4f}\n")

        f.write("\n📊 Confusion Matrix:\n")
        f.write(str(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"])))
        f.write("\n")

        f.write("\n📋 Classification Report:\n")
        f.write(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                      target_names=["No Error", "Has Error"]))


def baseline_detection_correction_CN_confidence(prefix_file, build_prompt, call_llms_api, parse_id_correction,
                                                sample_num, input_file="data/reviewed_data_CN_test.csv"):

    test_df = pd.read_csv(input_file)
    grouped = test_df.head(sample_num).groupby("Text ID", sort=False)
    results = []
    results_correct = []

    # Task : Error Detection --> Error Flag
    for inx, (text_id, group) in enumerate(tqdm(grouped)):
        full_text = group["Text"].tolist()[0]
        full_sentences = group["Sentences"].values[0].split('\r\n')
        gt_error_sentence = group["Error Sentence"].values[0]
        gt_error_sentence_id = group["Error Sentence ID"].values[0]
        gt_corrected_sentence = group["Corrected Sentence"].values[0]
        true_flag = group["Error Flag"].iloc[0]

        prompts = build_prompt(text_id, full_text)

        prompt = prompts[0]

        tolerance = 2
        while tolerance > 0:
            prediction = call_llms_api(prompt)

            if "NAN" in prediction:
                pred_flag = 0
                try:
                    pred_id, error_sent, correct_sent, detection_confidence, location_confidence, correction_confidence = parse_id_correction(prediction)
                except Exception as e:
                    tolerance -= 1
                    print("\nAgent error:", e, "  Tolerance:", tolerance)
                    continue

                results_correct.append({
                    "Text ID": text_id,
                    "Predicted Error Sentence": 'NAN',
                    "Predicted Error Sentence ID": -1,
                    "GroundTruth Error Sentence": gt_error_sentence,
                    "GroundTruth Error Sentence ID": gt_error_sentence_id,
                    "Predicted Corrected Sentence": 'NAN',
                    "GroundTruth Corrected Sentence": gt_corrected_sentence,
                    "detection_confidence": detection_confidence,
                    "location_confidence": location_confidence,
                    "correction_confidence": correction_confidence,
                })
                break
            else:
                pred_flag = 1
                try:
                    pred_id, error_sent, correct_sent, detection_confidence, location_confidence, correction_confidence = parse_id_correction(prediction)
                except Exception as e:
                    tolerance -= 1
                    print("\nAgent error:", e, "  Tolerance:", tolerance)
                    continue

                results_correct.append({
                    "Text ID": text_id,
                    "Predicted Error Sentence": pred_id,
                    "Predicted Error Sentence ID": pred_id,
                    "GroundTruth Error Sentence": gt_error_sentence,
                    "GroundTruth Error Sentence ID": gt_error_sentence_id,
                    "Predicted Corrected Sentence": correct_sent,
                    "GroundTruth Corrected Sentence": gt_corrected_sentence,
                    "detection_confidence": detection_confidence,
                    "location_confidence": location_confidence,
                    "correction_confidence": correction_confidence,
                })
                break

        if tolerance <= 0:
            pred_flag = 0
            results_correct.append({
                "Text ID": text_id,
                "Predicted Error Sentence": 'NAN',
                "Predicted Error Sentence ID": -1,
                "GroundTruth Error Sentence": gt_error_sentence,
                "GroundTruth Error Sentence ID": gt_error_sentence_id,
                "Predicted Corrected Sentence": 'NAN',
                "GroundTruth Corrected Sentence": gt_corrected_sentence,
                "detection_confidence": detection_confidence,
                "location_confidence": location_confidence,
                "correction_confidence": correction_confidence,
            })

        results.append(
            {"Text ID": text_id, "True Flag": true_flag, "Predicted Flag": pred_flag, "Output": prediction})
        print('\n', results[-1])

    results_correct_df = pd.DataFrame(results_correct)
    results_correct_df.to_csv(prefix_file + "_correction_results.csv", index=False)
    print("\n✅Saving multi-agent correction results:", prefix_file + "_correction_results.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(prefix_file + "_detection_results.csv", index=False)

    test_results = results_df[results_df["Predicted Flag"] != -1]
    accuracy = accuracy_score(test_results["True Flag"], test_results["Predicted Flag"])
    recall = recall_score(test_results["True Flag"], test_results["Predicted Flag"])
    print(f"\n✅ Error Detection Accuracy: {accuracy:.4f}")
    print(f"\n✅ Error Detection Recall: {recall:.4f}")

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"]))
    print("\n📋 Classification Report:")
    print(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                target_names=["No Error", "Has Error"]))

    with open(prefix_file + "_all_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(f"\n✅ Error Detection Accuracy: {accuracy:.4f}\n")
        f.write(f"\n✅ Error Detection Recall: {recall:.4f}\n")

        f.write("\n📊 Confusion Matrix:\n")
        f.write(str(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"])))
        f.write("\n")

        f.write("\n📋 Classification Report:\n")
        f.write(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                      target_names=["No Error", "Has Error"]))


def baseline_detection_correction_Arabic(prefix_file, build_prompt, call_llms_api, parse_id_correction, sample_num,
                                         input_file = "data/Multi_Lang/reviewed_data_ARA_test.csv"):

    test_df = pd.read_csv(input_file)
    grouped = test_df.head(sample_num).groupby("Text ID", sort=False)
    results = []
    results_correct = []

    # Task : Error Detection --> Error Flag
    for inx, (text_id, group) in enumerate(tqdm(grouped)):
        full_text = group["Text"].tolist()[0]
        full_sentences = group["Sentences"].values[0].split('\r\n')
        gt_error_sentence = group["Error Sentence"].values[0]
        gt_error_sentence_id = group["Error Sentence ID"].values[0]
        gt_corrected_sentence = group["Corrected Sentence"].values[0]
        true_flag = group["Error Flag"].iloc[0]

        prompts = build_prompt(text_id, full_text)

        prompt = prompts[0]

        tolerance = 2
        while tolerance > 0:
            prediction = call_llms_api(prompt)

            if "NAN" in prediction:
                pred_flag = 0
                results_correct.append({
                    "Text ID": text_id,
                    "Predicted Error Sentence": 'NAN',
                    "Predicted Error Sentence ID": -1,
                    "GroundTruth Error Sentence": gt_error_sentence,
                    "GroundTruth Error Sentence ID": gt_error_sentence_id,
                    "Predicted Corrected Sentence": 'NAN',
                    "GroundTruth Corrected Sentence": gt_corrected_sentence,
                })
                break
            else:
                pred_flag = 1
                try:
                    pred_id, error_sent, correct_sent = parse_id_correction(prediction)
                except Exception as e:
                    tolerance -= 1
                    print("\nAgent error:", e, "  Tolerance:", tolerance)
                    continue

                results_correct.append({
                    "Text ID": text_id,
                    "Predicted Error Sentence": pred_id,
                    "Predicted Error Sentence ID": pred_id,
                    "GroundTruth Error Sentence": gt_error_sentence,
                    "GroundTruth Error Sentence ID": gt_error_sentence_id,
                    "Predicted Corrected Sentence": correct_sent,
                    "GroundTruth Corrected Sentence": gt_corrected_sentence,
                })
                break

        if tolerance <= 0:
            pred_flag = 0
            results_correct.append({
                "Text ID": text_id,
                "Predicted Error Sentence": 'NAN',
                "Predicted Error Sentence ID": -1,
                "GroundTruth Error Sentence": gt_error_sentence,
                "GroundTruth Error Sentence ID": gt_error_sentence_id,
                "Predicted Corrected Sentence": 'NAN',
                "GroundTruth Corrected Sentence": gt_corrected_sentence,
            })

        results.append(
            {"Text ID": text_id, "True Flag": true_flag, "Predicted Flag": pred_flag, "Output": prediction})
        print('\n', results[-1])

    results_correct_df = pd.DataFrame(results_correct)
    results_correct_df.to_csv(prefix_file + "_correction_results.csv", index=False)
    print("\n✅Saving multi-agent correction results:", prefix_file + "_correction_results.csv")

    results_df = pd.DataFrame(results)
    results_df.to_csv(prefix_file + "_detection_results.csv", index=False)

    test_results = results_df[results_df["Predicted Flag"] != -1]
    accuracy = accuracy_score(test_results["True Flag"], test_results["Predicted Flag"])
    recall = recall_score(test_results["True Flag"], test_results["Predicted Flag"])
    print(f"\n✅ Error Detection Accuracy: {accuracy:.4f}")
    print(f"\n✅ Error Detection Recall: {recall:.4f}")

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"]))
    print("\n📋 Classification Report:")
    print(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                target_names=["No Error", "Has Error"]))

    with open(prefix_file + "_all_evaluation.txt", "w", encoding="utf-8") as f:
        f.write(f"\n✅ Error Detection Accuracy: {accuracy:.4f}\n")
        f.write(f"\n✅ Error Detection Recall: {recall:.4f}\n")

        f.write("\n📊 Confusion Matrix:\n")
        f.write(str(confusion_matrix(test_results["True Flag"], test_results["Predicted Flag"])))
        f.write("\n")

        f.write("\n📋 Classification Report:\n")
        f.write(classification_report(test_results["True Flag"], test_results["Predicted Flag"],
                                      target_names=["No Error", "Has Error"]))
