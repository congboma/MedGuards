import os
import re
from utils import (detect_MA_debate, get_system_provided_correct_na, locate_and_correct_with_multiagent,
                   locate_and_correct_with_multiagent_confidence, evaluate_correction_with_ids,
                   evaluate_correction_keywords)
from volcenginesdkarkruntime import Ark

# sample_num = 5
sample_num = 925

API_Key='api_key_here'
MODEL="doubao-1-5-thinking-pro-250415"
prefix_file = "results/doubao1.5_MedGuards"


# ========================== Step 1: Detection ==========================
def build_detection_prompt(text):
    formatted = text
    prompts = [
        f"""The following is a medical narrative about a patient. You are a skilled medical doctor reviewing the clinical text. The text is either correct or has at most a medical error related to treatment, management, cause, diagnosis or causalOrganism. Write down you thinking process in <think> thinking process here </think> tags. Check every sentence of the text. If the text is correct returns one word "CORRECT". If the text has a medical error returns one word "INCORRECT". Also output your confidence in <confidence>(1-100 score)</confidence> tags.\n\n{formatted}""",
        f"""You are a skilled medical doctor reviewing the clinical text. The text is either correct or has a medical error related to diagnosis (put more focus), treatment, management, cause or causalOrganism. Write down you thinking process in <think> thinking process here </think> tags. Check every sentence of the text. If the text is correct returns one word "CORRECT". If the text has a medical error returns one word "INCORRECT". Also output your confidence in <confidence>(1-100 score)</confidence> tags.\n\n{formatted}"""
    ]
    return prompts


def build_detection_debate_prompt(text):
    formatted = text
    prompts = [
        f"""You are a medical expert reviewing the clinical text and two junior doctors. The text is either accurate or contains at most a medical error, primarily related to diagnosis (pay closer attention), but may also involve treatment, management, cause, or causal organism. Document your reasoning within <think> your reasoning here </think> tags. Evaluate each sentence in the text. If the text is accurate, return the single word "CORRECT". If it contains a medical error, return the single word "INCORRECT". Also output your confidence in <confidence>(1-100 score)</confidence> tags.\n\n{formatted}"""]
    return prompts


# ========================== Step 2&3: Localization & Correction ==========================
def build_location_promptA(text, agent_name):
    return f"""You are {agent_name}, a medical report quality assurance expert. The report is either correct or has at most a medical error related to treatment, management, cause, diagnosis or causalOrganism. Your task is to identify the **one** sentence that contains an error (or there are zero error return 'NAN') in the following medical report and put your prediction in <result>Ensure you have results here</result> tags. Document your reasoning within <think></think> tags. A useful tip is that the error sentence (if there is) is more likely to be found in a conclusion sentence.

    Please ONLY return the original erroneous sentence exactly as it appears in the text (the text may also contain zero errors. If you are sure about it, please return 'NAN'). Also output your confidence in <confidence>(1-100 score)</confidence> tags.

    {text}
    """


def build_location_promptB(text, agent_name):
    return f"""You are {agent_name}, a medical report quality assurance expert. The report is either correct or has at most a medical error related to treatment, management, cause, diagnosis or causalOrganism. Your task is to identify the one sentence that contains an error (or there are zero error return 'NAN') in the following medical report. A useful tip is that the error sentence (if there is) is more likely to be found in a conclusion sentence. Put your prediction in <result>Ensure you have results here</result> tags and document your reasoning within <think></think> tags. Also output your confidence in <confidence>(1-100 score)</confidence> tags.

    If the text has an error, output:
    <result>The Error Sentence</result>

    If the text has no error, output:
    NAN

    {text}
    """


def build_location_discuss_prompt(agent_name, full_text, partner_opinion):
    return f"""You are a senior medical report quality assurance expert. The report is either correct or has at most a medical error related to treatment, management, cause, diagnosis or causalOrganism.

    Here is the full report:
    {full_text}

    Your 2 colleagues suggest that the following sentence is erroneous with reasons:

    "{partner_opinion}"

    Based on the above information, please provide the ONLY error sentence **you** believe (or there are zero error return 'NAN'). Please put results in <result>Ensure you have results here</result> tags and document your reasoning within <think></think> tags. A useful tip is that the error sentence (if there is) is more likely to be found in a conclusion sentence. Also output your confidence in <confidence>(1-100 score)</confidence> tags.
    """


def build_correction_prompt(full_text, error_sentence):
    return f"""You are a medical report quality assurance expert. The following sentence is incorrect and has a medical error related to treatment, management, cause, diagnosis or causalOrganism. Please provide a corrected version to <result>the corrected sentence</result> tags (your thinking reason can be provided in <think></think> tags). Also, ensure you output results between <result></result> tags and output your confidence in <confidence>(1-100 score)</confidence> tags.

    Here is the original full report:
    {full_text}

    Predicted Erroneous Sentence:
    {error_sentence}

    Corrected Sentence:"""


# ========================== End of prompting ==========================


def extract_think_content(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_result(text):
    if "INCORRECT" in text.upper():
        return "INCORRECT"
    elif "CORRECT" in text.upper():
        return "CORRECT"
    else:
        return "UNKNOWN"


def extract_confidence_from_text(text):
    match = re.search(r"<confidence>(\d{1,3})</confidence>", text)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 100:
            return val
    return None


def extract_result_loc_corr(text):
    match = re.search(r"<result>(.*?)</result>", text)
    return match.group(1).strip() if match else None


def call_doubao_api_with_thinking(prompt):
    try:
        client = Ark(api_key=API_Key)

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        message = completion.choices[0].message.content.strip()

        return {
            "output": message,
            "result": extract_result(message),
            "think": extract_think_content(message),
            "confidence": extract_confidence_from_text(message)
        }

    except Exception as e:
        print("API Error：", e)
        return {
            "output": "ERROR",
            "result": "UNKNOWN",
            "think": None,
            "confidence": None
        }


def call_doubao_api_loc_corr_thinking(prompt):
    try:
        client = Ark(api_key=API_Key)

        completion = client.chat.completions.create(
            model="doubao-1-5-thinking-pro-250415",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        message = completion.choices[0].message.content.strip()

        return {
            "output": message,
            "result": extract_result_loc_corr(message),
            "think": extract_think_content(message),
            "confidence": extract_confidence_from_text(message)
        }

    except Exception as e:
        print("API Error：", e)
        return {
            "output": "ERROR",
            "result": "UNKNOWN",
            "think": None,
            "confidence": None
        }


def call_doubao_api(prompt):
    try:
        client = Ark(api_key=API_Key)

        completion = client.chat.completions.create(
            model="doubao-1-5-thinking-pro-250415",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        message = completion.choices[0].message.content.strip()
        return message
    except Exception as e:
        print("API Error：", e)
        return "ERROR"


if __name__ == "__main__":
    detect_MA_debate(prefix_file, build_detection_prompt, build_detection_debate_prompt, call_doubao_api_with_thinking, sample_num)

    locate_and_correct_with_multiagent_confidence(prefix_file, build_location_promptA, call_doubao_api_loc_corr_thinking,
                                       build_location_discuss_prompt, build_correction_prompt, sample_num,
                                       build_location_promptB)

    count = get_system_provided_correct_na(prefix_file)
    counters = {"total_texts": sample_num, "system_provided_correct_na": count}
    evaluate_correction_with_ids(prefix_file, counters, sample_num)

    evaluate_correction_keywords(prefix_file, counters, sample_num)
