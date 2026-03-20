import os, yaml, json, copy
import subprocess
from PIL import Image
from tqdm  import tqdm

Action_tokens = {
    "region_x": "<|x_0|>,<|x_1|>,<|x_2|>,<|x_3|>,<|x_4|>,<|x_5|>,<|x_6|>,<|x_7|>".split(","),
    "region_y": "<|y_0|>,<|y_1|>,<|y_2|>,<|y_3|>,<|y_4|>,<|y_5|>,<|y_6|>,<|y_7|>".split(","),
    "dino": "<|detection_action_start|>",
    "clip": "<|clip_action_start|>",
    "sam": "<|seg_action_start|>",
}

os.chdir("../modules/LLaMA-Factory")

def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data



def generate_dino_item(gt_item, pred_item, index, use_answers = False):
    image_path = gt_item["images"][0]
    new_message = [
        
        {
            "content": gt_item["messages"][0]["content"].replace('Require additional perception features, and then answer the question.', 'Let\'s think step by step.'), 
            "role": "user"
        },
        {
            "content": "<|start-latent|>"+"<|latent|>"*8+"<|end-latent|>",
            "role": "assistant"
        },
        {
            "content": "Require additional perception features if required and then answer the question based on your observations or answer the question directly if additional perception features are not required", 
            "role": "user"
        },
        {"role": "assistant", "content": "<|detection_action_start|><|detection_action|><|detection_action_end|>"},
        {"role": "user", "content": "<detection_image>"},
        gt_item["messages"][-1],
    ]
    new_item = {
        "index": index,
        "messages": new_message,
        "images": [image_path],
        "detection_images": [image_path],
        "clip_images": [],
        'seg_images': [],
        'annotations': gt_item['annotations'] or [],
        # 'episode_id': gt_item['episode_id'],
        # 'step_id': gt_item['step_id']
    }
    
    return new_item


def generate_third_round_data_items(original_data_file_path, answer_2r_file_path, answer_1r_file_path, use_answers = False):
    data = json.load(open(original_data_file_path, "r"))
    answer_2r = [json.loads(l) for l in open(answer_2r_file_path, "r").readlines()]
    answer_1r = [json.loads(l) for l in open(answer_1r_file_path, "r").readlines()]

    new_data = []
    finished_data = []

    for i, (gt_item, generated_item, round_1_answers) in enumerate(zip(data[:], answer_2r[:], answer_1r[:])):
        geneated_answer = generated_item["predict"]
        do_dinoo = Action_tokens["dino"] in geneated_answer
        # do_dinoo = True
        
        if do_dinoo:
            new_item = generate_dino_item(gt_item, round_1_answers, i, use_answers = False)
            new_data.append(new_item)
            # finished_item = None
        else:
            
            # print("Neither dino nor region tokens found in the answer")
            finished_data.append({
                "index": i,
                "question": generated_item['prompt'],
                "predict": geneated_answer,
                "label": gt_item["messages"][-1]["content"].replace("Now answer the question.\n", ""),
                "images": gt_item["images"]
            })
        # if i == 100:
        #     break
    
    return new_data, finished_data



def generate_second_round_data(original_data_file_path, answer_1r_file_path, use_answers = False):
    data = json.load(open(original_data_file_path, "r"))
    answer_1r = [json.loads(l) for l in open(answer_1r_file_path, "r").readlines()]
    new_data = []
    for i, (meta_item, meta_pred_item) in enumerate(tqdm(zip(data[:], answer_1r[:]))):
        messages = [
            {
                "content": meta_item["messages"][0]["content"].replace('Require additional perception features, and then answer the question.', 'Let\'s think step by step.'), 
                "role": "user"
            },
            {
                "content": "<|start-latent|>"+"<|latent|>"*8+"<|end-latent|>",
                "role": "assistant"
            },
            # {
            #     "content": "Require additional perception features, and then answer the question based on your observations.", 
            #     "role": "user"
            # },
            
            {
                "content": "Require additional perception features if required and then answer the question based on your observations or answer the question directly if additional perception features are not required", 
                "role": "user"
            },
            {
                'content': "",
                "role": "assistant"
            }
        ]
        new_item = {
            "index": i,
            "messages": messages,
            "images": meta_item["images"][:1],
            "detection_images": [],   
            "clip_images": [],
            'seg_images': [],
            'annotations': meta_item.get('annotations', []),
            # 'episode_id': meta_item['episode_id'],
            # 'step_id': meta_item['step_id']
        }
        new_data.append(new_item)
        # if i == 100:
        #     break
        
        
    return new_data


def generate_first_round_data_items(original_data_file_path):
    data = json.load(open(original_data_file_path, "r"))
    new_data = []
    for i, meta_item in enumerate(tqdm(data[:])):
        messages = [
            {
                "content": meta_item["messages"][0]["content"].replace('Require additional perception features, and then answer the question.', 'Let\'s think step by step.'), 
                "role": "user"
            },
            {
                "content": "",
                "role": "assistant"
            },
        ]
        new_item = {
            "index": i,
            "messages": messages,
            "images": meta_item["images"][:1],
            "detection_images": [],
            "clip_images": [],
            'seg_images': [],
            'annotations': meta_item.get('annotations', []),
            # 'episode_id': meta_item['episode_id'],
            # 'step_id': meta_item['step_id']
        }
        new_data.append(new_item)
    return new_data



base_yaml_file = "inference/evaluation.yaml"
base_parameters = read_yaml(base_yaml_file)

temp_dataset_name = "ishift_evaluation"
temp_dataset_file = "evaluation.json"

if __name__ == "__main__":

    datasets = ['train']

    models = {
        "models/iSHIFT":{
            "num_inner_forward_run":2,
            "vision_encoder_ls":"dino",
        },
    }
    print(datasets)
    
    p_val,k_val = 0.1,0
    # use_answers = False
    previous_model_path = None
    previous_dataset_name = None
    for use_answers in [True]:
        for dataset_name in datasets:
            for model_path, model_para in models.items():

                while not os.path.exists(model_path):
                    # print(f"Model path {model_path} does not exist. Skipping this model.")
                    print('Checking for checkpoint')
                    import time
                    time.sleep(600)
                    continue
                print(f"Model path {model_path} exists")
                print(dataset_name)
                # overall config
                # output_dir = os.path.join(model_path, dataset_name, f"p{p_val}_k{k_val}")
                output_dir = os.path.join(model_path, dataset_name, 'use_answers' if use_answers else 'no_use_answers')
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(output_dir+ "/round1", exist_ok=True)
                os.makedirs(output_dir+ "/round2", exist_ok=True)
                os.makedirs(output_dir+ "/round3", exist_ok=True)

                if os.path.exists(output_dir+"/final_scores.txt"):
                    with open(output_dir+"/final_scores.txt", "r") as score_file:
                        score_str = score_file.read()
                    print(output_dir+"/final_scores.txt Finished. ", score_str)
                    continue
                
                dataset_file = f"sample_dataset/{dataset_name}.json"

                image_resolution = 512
                # data 1r
                data_1r = generate_first_round_data_items(dataset_file)
                json.dump(
                    data_1r, 
                    open(temp_dataset_file, "w"),
                    indent=2
                )

                # generation 1r
                output_file_1r = output_dir + "/round1/generated_predictions.jsonl"
                if not os.path.exists(output_file_1r):
                    print(output_file_1r)
                    print(f"Round 1")
                    
                    parameters_copy = copy.deepcopy(base_parameters)
                    parameters_copy["output_dir"] = output_dir + "/round1"
                    parameters_copy["model_name_or_path"] = model_path
                    parameters_copy["eval_dataset"] = temp_dataset_name
                    parameters_copy["image_resolution"] = image_resolution
                    parameters_copy["top_k"] = k_val
                    parameters_copy["top_p"] = p_val
                    
                    for k,v in model_para.items():
                        if v is not None:
                            parameters_copy[k] = v

                    log_file = output_dir+f"/round1/generation.log"

                    command = "cd ../modules/LLaMA-Factory; llamafactory-cli train "
                    for k,v in parameters_copy.items():
                        command = command + f"--{k} {v} "
                    command = command + f" > {log_file} 2>&1"
                    if not os.path.exists(output_dir + "/round1/generated_predictions.jsonl"):
                        subprocess.run(command, shell=True, check=True)

                # data 2r
                # read the output file
                
                print('Round 2')
                data_2r = generate_second_round_data(dataset_file, output_file_1r, use_answers=use_answers)

                
                output_file_2r = output_dir + "/round2/generated_predictions.jsonl" 
                # import shutil
                # shutil.copy(output_file_1r, output_file_2r)

                if len(data_2r) == 0:
                    print('no round 2')
                    pass
                else:
                    if not os.path.exists(output_file_2r):
                        json.dump(
                            data_2r, 
                            open(temp_dataset_file, "w"),
                            indent=2
                        )
                        # break
                        # generation 2r
                        parameters_copy = copy.deepcopy(base_parameters)
                        parameters_copy["output_dir"] = output_dir + "/round2"
                        parameters_copy["model_name_or_path"] = model_path
                        parameters_copy["eval_dataset"] = temp_dataset_name
                        parameters_copy["image_resolution"] = image_resolution
                        parameters_copy["top_k"] = k_val
                        parameters_copy["top_p"] = p_val
                        parameters_copy["per_device_eval_batch_size"] = 25
                        for k,v in model_para.items():
                            if v is not None:
                                parameters_copy[k] = v
                                
                        log_file = output_dir+f"/round2/generation.log"

                        command = "cd ../modules/LLaMA-Factory; llamafactory-cli train "
                        for k,v in parameters_copy.items():
                            command = command + f"--{k} {v} "
                        command = command + f" > {log_file} 2>&1"
                        if not os.path.exists(output_dir + "/round2/generated_predictions.jsonl"):
                            subprocess.run(command, shell=True, check=True)

                
                

                print('Round 3')
                data_3r, final_answers = generate_third_round_data_items(dataset_file, output_file_2r, output_file_1r, use_answers=use_answers)

                with open(output_dir + '/round3/fast_answers.jsonl', 'w') as f:
                    for item in final_answers:
                        f.write(json.dumps(item) + "\n")

                if len(data_3r) == 0:
                    print('no round 3')
                    pass
                else:
                    json.dump(
                        data_3r, 
                        open(temp_dataset_file, "w"),
                        indent=2
                    )
                    # break
                    # generation 3r
                    parameters_copy = copy.deepcopy(base_parameters)
                    parameters_copy["output_dir"] = output_dir + "/round3"
                    parameters_copy["model_name_or_path"] = model_path
                    parameters_copy["eval_dataset"] = temp_dataset_name
                    parameters_copy["image_resolution"] = image_resolution
                    parameters_copy["top_k"] = k_val
                    parameters_copy["top_p"] = p_val
                    parameters_copy["per_device_eval_batch_size"] = 25
                    for k,v in model_para.items():
                        if v is not None:
                            parameters_copy[k] = v
                            
                    log_file = output_dir+f"/round3/generation.log"

                    command = "cd ../modules/LLaMA-Factory; llamafactory-cli train "
                    for k,v in parameters_copy.items():
                        command = command + f"--{k} {v} "
                    command = command + f" > {log_file} 2>&1"
                    if not os.path.exists(output_dir + "/round3/generated_predictions.jsonl"):
                        subprocess.run(command, shell=True, check=True)

                # evaluation
                output_file_3r = output_dir + "/round3/generated_predictions.jsonl" 
                if os.path.exists(output_file_3r):
                    with open(dataset_file, "r") as gt_data_file:
                        gt_data = json.load(gt_data_file)
                        with open(output_file_3r, "r") as output_file:
                            for i, (data_3r_item, generated_line) in enumerate(zip(data_3r, output_file.readlines())):
                                question_index = data_3r_item["index"]
                                gt_item = gt_data[question_index]

                                generated_item = json.loads(generated_line)

                                # check_question(gt_item, generated_item)

                                answer = generated_item["predict"]

                                # final_answers.append({
                                #     "index": question_index,
                                #     "question": gt_item["messages"][0]["content"],
                                #     "answer": answer,
                                #     "ground_truth": gt_item["messages"][-1]["content"].replace("Now answer the question.\n", ""),
                                #     "images": gt_item["images"]
                                # })

                fast_responses = final_answers
                if os.path.exists(output_file_3r):
                    with open(output_file_3r, 'r') as f:
                        slow_responses = [json.loads(line) for line in f.readlines()]
                else:
                    slow_responses = []

                for r in fast_responses:
                    slow_responses.insert(r['index'], r)
                
                # if not os.path.exists(output_dir + '/round3/' + 'final_predictions.jsonl'):
                with open(output_dir + '/round3/' + 'final_predictions.jsonl', 'w') as f:
                    for item in slow_responses:
                        f.write(json.dumps(item) + "\n")

                previous_model_path = model_path
                previous_dataset_name = dataset_name
            
