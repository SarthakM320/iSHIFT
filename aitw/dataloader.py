import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import argparse
from tqdm import tqdm
from PIL import Image
from os.path import join
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import action_matching, action_type
import numpy as np
import random
import re

def load_data(args, split):
    target_text = []
    source_text = []
    source_image = []
    anno_positions = []
    episode_ids = []
    step_ids = []
    slow_responses = []

    if args.all_data:
        if split == "train" or split == 'val':
            data = []
            # , "install", 
            for subdir in ["install", "general", "single", "google_apps", "web_shopping"]:
                # print(f"loading {subdir}", len(data))
                with open(f"{args.data_root}/{subdir}/{split}.obj", "rb") as rp:
                    sub_data = pickle.load(rp)
                # if subdir == "google_apps":
                #     sub_data = random.sample(sub_data, int(len(sub_data) * args.all_data))
                print(subdir, len(sub_data))
                data.extend(sub_data)
        else:
            # we use general subset for dev/test
            with open(f"{args.data_root}/{args.eval_subset}/test.obj", "rb") as rp:
                data = pickle.load(rp)
                print(len(data))
                data = data[:128]
            if args.eval_subset == "google_apps":
                print(len(data))
                data = random.sample(data, int(len(data) * args.all_data))
    else:
        with open(f"{args.data_root}_{split}.obj", "rb") as rp:
            data = pickle.load(rp)
            if args.data_ratio:
                data = random.sample(data, int(len(data) * args.data_ratio))

    for qid, episode in enumerate(tqdm(data)):
        episode_id = episode["episode_id"]
        episode_data = episode["data"]
        if args.use_history:
            history_action = []
            if args.use_img_history:
                # history_image = [torch.zeros(args.img_dim)] * args.use_history
                history_image = []

        for step_idx, step_data in enumerate(episode_data):
            
            question = step_data["goal"]
            question = f"Goal: {question}"

            image_path = join(args.image_dir, step_data["image_path"])
            # image = Image.open(image_path)
            image = image_path

            ui_positions = step_data["ui_positions"]
            ui_text = step_data["ui_text"]
            ui_type = step_data["ui_type"]

            if args.use_layout:
                icon_string = ""
                for ui_idx, ui_type_i in enumerate(ui_type):
                    ui_axis = ui_positions[ui_idx]
                    top, left, height, width = ui_axis
                    # The y-axis is inverted for AndroidEnv, so bottom = top + height.
                    bottom, right = top + height, left + width
                    ui_axis = [top, left, bottom, right]
                    ui_axis = ["{:.4f}".format(axis) for axis in ui_axis]
                    ui_axis = f"({ui_axis[0]}, {ui_axis[1]}, {ui_axis[2]}, {ui_axis[3]})"
                    if ui_type_i == "TEXT":
                        icon_string += f'<p id={ui_idx} class="text" alt="{ui_axis}">{ui_text[ui_idx]}</p>\n'
                    elif "ICON" in ui_type_i:
                        icon_string += f'<img id={ui_idx} class={ui_type_i} alt="{ui_axis}">{ui_text[ui_idx]}</p>\n'
                    else:
                        print(icon_string)
                        assert "parsing ui failed!!!"
                
                question = f"{question}\nScreen: {icon_string}"
                # print(question)
            result_touch_yx = step_data["result_touch_yx"]
            result_lift_yx = step_data["result_lift_yx"]
            result_action = step_data["result_action"][0]
            result_text = step_data["result_action"][1]

            result_text = result_text.replace("\\", "").replace('"','').replace("'","")

            # Slow vs fast: slow = needs precise prediction (tap, TYPE, etc.); fast = scroll, PRESS_BACK, etc.
            action_touch_yx = np.asarray(result_touch_yx, dtype=float)
            action_lift_yx = np.asarray(result_lift_yx, dtype=float)
            
            if result_action == "DUAL_POINT":
                is_slow = is_tap_action(action_touch_yx, action_lift_yx)
            else:
                is_slow = False
            slow_responses.append(is_slow)

            if args.transform_axis:
                scroll_map = {
                    "up": [[0.8000, 0.5000], [0.2000, 0.5000]],
                    "down": [[0.2000, 0.5000], [0.8000, 0.5000]],
                    "left": [[0.5000, 0.8000], [0.5000, 0.2000]],
                    "right": [[0.5000, 0.2000], [0.5000, 0.8000]]
                }
                if result_action == "DUAL_POINT":
                    if is_tap_action(action_touch_yx, action_lift_yx):
                        result_touch_yx = [round(axis, 4) for axis in result_touch_yx]
                        # if touching, the lift can be the same as touch
                        result_lift_yx = result_touch_yx
                    else:
                        drags_match = _check_drag_actions_match(
                            action_touch_yx, action_lift_yx
                        )
                        result_touch_yx, result_lift_yx = scroll_map[drags_match]

            target_action = f'"action_type": "{result_action}", "touch_point": "{result_touch_yx}", "lift_point": "{result_lift_yx}", "typed_text": "{result_text}"'
            
            if args.use_history:
                prev_actions = "\n".join(history_action)
                question = f"Previous Actions: {prev_actions}\n{question}"
                if args.use_img_history:
                    image = history_image + [image]
                    image = torch.stack(image)

            if args.use_future:
                future_actions = episode_data[step_idx:]
                if len(future_actions) > args.use_future:
                    future_actions = future_actions[:args.use_future]
                future_actions = "[" + ",".join([action_t["result_action"][0] for action_t in future_actions]) + "]"
                target_action_label = "Action Plan: " + future_actions + "; Action Decision: " + target_action

            source_text.append(question)
            source_image.append(image)
            target_text.append(target_action_label)
            anno_positions.append(ui_positions)
            episode_ids.append(episode_id)
            step_ids.append(step_idx)

            if args.use_history:
                history_action.append(target_action)
                if args.use_img_history:
                    history_image.append(image[-1])
                    history_image.pop(0)
                if len(history_action) > args.use_history:
                    history_action.pop(0)
                        

        if args.debug_num:
            if int(qid) > args.debug_num:
                break

    return source_text, source_image, target_text, anno_positions, episode_ids, step_ids, slow_responses

_SWIPE_DISTANCE_THRESHOLD = 0.04
def is_tap_action(normalized_start_yx, normalized_end_yx):
    distance = np.linalg.norm(
        np.array(normalized_start_yx) - np.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD

def _check_drag_actions_match(
    drag_touch_yx,
    drag_lift_yx,
):
    """Determines if two drag actions are the same."""
    # Store drag deltas (the change in the y and x coordinates from touch to
    # lift), magnitudes, and the index of the main axis, which is the axis with
    # the greatest change in coordinate value (e.g. a drag starting at (0, 0) and
    # ending at (0.3, 0.5) has a main axis index of 1).
    drag_1_deltas = drag_lift_yx - drag_touch_yx
    drag_1_magnitudes = np.abs(drag_1_deltas)
    drag_1_main_axis = np.argmax(drag_1_magnitudes)

    # y axis
    if drag_1_main_axis == 0:
        if drag_1_deltas[0] < 0:
            scroll = "up"
        else:
            scroll = "down"
    elif drag_1_main_axis == 1:
        if drag_1_deltas[1] < 0:
            scroll = "left"
        else:
            scroll = "right"
            
    return scroll

def load_data_sharegpt(args, split):
    conversations = []

    source_text, source_image, target_text, anno_positions, episode_ids, step_ids, slow_responses = load_data(args, split)

    for i in range(len(source_text)):
        conversation = {
            "messages": [
                {
                    "content": f"<image>\n{source_text[i]}\nPredict the next action to be taken according to the Goal\nRequire additional perception features, and then answer the question.",
                    "role": "user"
                },
                { 
                    "content": "<|detection_action_start|><|detection_action|><|detection_action_end|>", 
                    "role": "assistant"
                },
                { 
                    "content": "<detection_image>", 
                    "role": "user" 
                },
                {
                    "content": target_text[i],
                    "role": "assistant"
                }
            ],
            "images": [source_image[i]],
            "detection_images": [source_image[i]],
            "annotations": anno_positions[i].tolist(),
            "episode_id": episode_ids[i],
            "step_id": step_ids[i],
            "slow_response": slow_responses[i],
        }
        # [ { "content": "What type of structure does the clock tower belong to in the image?\n<image>\nRequire additional perception features, and then answer the question.", "role": "user" }, { "content": "<|detection_action_start|><|detection_action|><|detection_action_end|>", "role": "assistant" }, { "content": "<detection_image>", "role": "user" }, { "content": "The clock tower belongs to a church, as the image shows the large tower with a clock on it as a part of a church.", "role": "assistant" } ]
        conversations.append(conversation)

    return conversations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument('--eval_subset', type=str, default='google_apps')
    parser.add_argument('--all_data', type=float, default=0.03)
    parser.add_argument('--data_ratio', type=float, default=None)
    parser.add_argument('--use_history', type=int, default=2)
    parser.add_argument('--use_img_history', action='store_true')
    parser.add_argument('--img_dim', type=int, default=768)
    parser.add_argument('--use_layout', action='store_true')
    parser.add_argument('--transform_axis', action='store_true')
    parser.add_argument('--use_future', type=int, default=1) # TODO
    parser.add_argument('--debug_num', type=int, default=None)
    
    args = parser.parse_args()
    args.image_dir = 'data/images'
    
    # Test loading training data
    # source_text, source_image, target_text, anno_positions = load_data(args, "train")
    subset = "test"
    import json
    if subset != 'test':
        conversations = load_data_sharegpt(args, subset)
        with open(f'sharegpt/{subset}_all_2.json', 'w') as f:
            json.dump(conversations, f)
    else:
        for name in ["general", "web_shopping"]:
            args.eval_subset = name
            conversations = load_data_sharegpt(args, subset)
            print(len(conversations))
            with open(f'sharegpt/{subset}_{args.eval_subset}_128.json', 'w') as f:
                json.dump(conversations, f)
    # import random
    # # idx = random.randint(0, len(source_text)-1)
    # idx = 85
    # print(idx)
    # print(f"Loaded {len(source_text)} training examples")
    # print("\nSample source text:", source_text[idx])
    # print("\nSample target text:", target_text[idx])
    # print("\nSample image shape:", source_image[idx].size)
    # # print("\nSample UI positions:", anno_positions[idx])

if __name__ == "__main__":
    main()