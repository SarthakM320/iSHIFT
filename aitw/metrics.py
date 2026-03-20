import os
import re
import ast
import json
import numpy as np
from tqdm import tqdm
from utils import save_json, AverageMeter, ProgressMeter, Summary, dict_to_cuda
from action_matching import  check_actions_match, is_tap_action

NUM_HISTORY = 0

import logging
logging.basicConfig(level=logging.INFO)


def calculate_aitw_metrics(results):
    corr_action=0
    corr_type=0
    num_text=0
    corr_text=0
    num_scroll=0
    corr_scroll=0
    num_click=0
    corr_click=0
    num_both_click=0
    corr_both_click=0
    num_wrong_format=0
    num=0
    for output in results:
        corr_action += output['corr_action']
        corr_type += output['corr_type']
        num_text += output['num_text']
        corr_text += output['corr_text']
        num_scroll += output['num_scroll']
        corr_scroll += output['corr_scroll']
        num_click += output['num_click']
        corr_click += output['corr_click']
        num_both_click += output['num_both_click']
        corr_both_click += output['corr_both_click']
        num_wrong_format += output['num_wrong_format']
        num += 1

    logging.info("[Score]: " + str(corr_action/num))
    logging.info("[Valid]: " + str(num_wrong_format/num))
    metrics = {
        "Score": corr_action / num,
        "Num Corr Action": corr_action,
        "Num Corr Type": corr_type,

        "Num Text": num_text,
        "Num Corr Text": corr_text,

        "Num Scroll": num_scroll,
        "Num Corr Scroll": corr_scroll,

        "Num Click": num_click,
        "Num Corr Click": corr_click,

        "Num Both Click": num_both_click,
        "Num Corr Both Click": corr_both_click,

        "Num Wrong Format": num_wrong_format,
        "Num": num,
    }
    return metrics

import enum
class ActionType(enum.IntEnum):
    UNUSED_0 = 0
    UNUSED_1 = 1
    UNUSED_2 = 2
    UNUSED_8 = 8
    UNUSED_9 = 9
    TYPE = 3
    DUAL_POINT = 4
    PRESS_BACK = 5
    PRESS_HOME = 6
    PRESS_ENTER = 7
    STATUS_TASK_COMPLETE = 10
    STATUS_TASK_IMPOSSIBLE = 11

def action_to_json(text):
    action_plan_match = re.search(r'Action Plan:\s*\[(.*?)\]', text)
    action_plan = action_plan_match.group(1) if action_plan_match else None

    action_decision_match = re.search(r'Action Decision:\s*(.*)', text)
    action_decision_str = action_decision_match.group(1) if action_decision_match else ""

    action_decision_dict = {}
    matches = re.findall(r'"(.*?)":\s*"([^"]*?)"', action_decision_str)

    for key, value in matches:
        if "[" in value and "]" in value:
            value = json.loads(value)
        elif key == "action_type":
            value = ActionType[value.upper()].value if value.upper() in ActionType.__members__ else value
        action_decision_dict[key] = value

    return action_decision_dict

def validate_aitw(file_path, dataset_file_path):

    answers_unique = []
    generated_texts_unique = []
    outputs_unique = []

    metric = 0

    generated = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                generated.append(json.loads(line))
            except:
                print(line)

    with open(dataset_file_path, 'r') as dataset_file:
        dataset = json.load(dataset_file)

    for i, (ori, gen) in enumerate(zip(dataset,generated)):

        outputs = {
            "sentence": gen['output'],
            "label": gen['gt'],
            "annot_position": ori['annotations'],
            "ep_id": ori['episode_id'],
        }

        outputs.update(dict(
            corr_action=0,
            corr_type=0,
            num_text=0,
            corr_text=0,
            num_scroll=0,
            corr_scroll=0,
            num_click=0,
            corr_click=0,
            num_both_click=0,
            corr_both_click=0,
            num_wrong_format=0,
        ))

        generated_texts_unique.append(gen['output'])
        answers_unique.append(gen['gt'])
        outputs_unique.append(outputs)

    domain = split
    checkpoint = 'ishift'

    results = {}
    for pred_i, ans_i, output_i in tqdm(zip(generated_texts_unique, answers_unique, outputs_unique)):
        if domain not in results:
            results[domain] = []

        try:
            action_pred = action_to_json(pred_i)
            action_ref = action_to_json(ans_i)
        except Exception as e:
            continue

        annot_position = output_i["annot_position"]
        try:
            check_match = check_actions_match(action_pred["touch_point"],
                                                            action_pred["lift_point"],
                                                            action_pred["action_type"],
                                                            action_ref["touch_point"],
                                                            action_ref["lift_point"],
                                                            action_ref["action_type"],
                                                            np.array(annot_position))
        except Exception as e:
            output_i['num_wrong_format'] += 1
            check_match = False

        if check_match == True:
            output_i['corr_action'] += 1
            match_label = 1
        else:
            match_label = 0

        try:
            if action_pred["action_type"] == action_ref["action_type"]:
                output_i['corr_type'] += 1

            if action_ref["action_type"] == 3:
                output_i['num_text'] += 1
                if (action_pred["typed_text"] == action_ref["typed_text"]) or (
                        action_pred["typed_text"] in action_ref["typed_text"]) or (
                        action_ref["typed_text"] in action_pred["typed_text"]):
                    output_i['corr_text'] += 1

            if action_ref["action_type"] == 4:
                if is_tap_action(action_ref["touch_point"], action_ref["lift_point"]):
                    output_i['num_click'] += 1
                    if match_label:
                        output_i['corr_click'] += 1
                else:
                    output_i['num_scroll'] += 1
                    if match_label:
                        output_i['corr_scroll'] += 1
                if (action_pred["action_type"] == 4) and is_tap_action(action_ref["touch_point"],
                                                                        action_ref[
                                                                            "lift_point"]) and is_tap_action(
                        action_pred["touch_point"], action_pred["lift_point"]):
                    output_i['num_both_click'] += 1
                    if match_label:
                        output_i['corr_both_click'] += 1

        except Exception as e:
            output_i['num_wrong_format'] += 1

        results[domain].append(output_i)

    eval_dict = {}
    for domain in results.keys():
        logging.info("==="*10)
        logging.info(f"Domain: {domain}")
        logging.info(f"Checkpoint: {checkpoint}")
        logging.info("==="*10)
        eval_dict[domain] = calculate_aitw_metrics(results[domain])

    metric = sum([x["Score"] for x in eval_dict.values()]) / len(eval_dict)
    logging.info("==="*10)
    logging.info(f"[Avg Score]: {metric}")
    logging.info("==="*10)

    print(file_path)
    print(eval_dict)

    return metric

if __name__ == "__main__":
    for split in ['test_general', 'test_single', 'test_install', 'test_google_apps', 'test_web_shopping']:
        try:
            validate_aitw(
                f'inference/results/{split}/results_all.jsonl',
                f'sample_dataset/{split}.json'
            )
        except Exception as e:
            print(f"Error in {split}: {e}")
            continue
