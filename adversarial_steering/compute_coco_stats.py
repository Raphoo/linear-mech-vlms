import os
from os import path
import torch
import numpy as np
from utils.coco import load_coco_annotations, get_subject_spatial_id
from utils.linalg import project_onto_plane, euclidean_distances
from utils.extract_embeds import VLMForExtraction
from transformers import AutoProcessor
import argparse
import json
import traceback

def compute_verdict(data_dir, tokenizer, model, postfix=None):
    results = {}
    for fname in os.listdir(data_dir):
        try:
            # fig, ax = plt.subplots(1, 1, figsize=(12, 6.2))
            object_a, object_b, pos_word, neg_word, id = fname.split("_")
            object_a, object_b = object_a.replace("+", " "), object_b.replace("+", " ")
            id = int(id)

            if postfix is None:
                sequences = torch.load(
                    path.join(data_dir, fname, f"sequences.pt"),
                    weights_only=False,
                    map_location="cpu",
                )
                logits = torch.load(
                    path.join(data_dir, fname, f"logits.pt"),
                    weights_only=False,
                    map_location="cpu",
                )
            else:
                sequences = torch.load(
                    path.join(data_dir, fname, f"sequences_{postfix}.pt"),
                    weights_only=False,
                    map_location="cpu",
                )
                logits = torch.load(
                    path.join(data_dir, fname, f"logits_{postfix}.pt"),
                    weights_only=False,
                    map_location="cpu",
                )
            response = (
                tokenizer.decode(sequences[0])
                .split("assistant")[-1]
                .split("ASSISTANT")[-1]
                .split("<start_of_turn>model")[-1]
                .lower()
            )
            sequences = sequences[0][-len(logits) :]

            if pos_word in response and neg_word not in response:
                verdict = "correct"
            elif neg_word in response and pos_word not in response:
                verdict = "wrong"
            else:
                verdict = "nonsense"

            log_prob = torch.log_softmax(logits[0], dim=-1).squeeze()
            if "llava" in model:
                pos_tokens = [tokenizer.tokenize(": " + pos_word.capitalize())[-1]]
                neg_tokens = [tokenizer.tokenize(": " + neg_word.capitalize())[-1]]
            elif "llama" in model:
                pos_tokens = [
                    tokenizer.tokenize(tw)[-1]
                    for tw in [pos_word, pos_word.capitalize()]
                ]
                neg_tokens = [
                    tokenizer.tokenize(tw)[-1]
                    for tw in [neg_word, neg_word.capitalize()]
                ]
            else:
                pos_tokens = [tokenizer.tokenize(pos_word)[-1]]
                neg_tokens = [tokenizer.tokenize(neg_word)[-1]]

            pos_ids, neg_ids = tokenizer.convert_tokens_to_ids(
                pos_tokens
            ), tokenizer.convert_tokens_to_ids(neg_tokens)
            pos_log_prob = log_prob[pos_ids].max().item()
            neg_log_prob = log_prob[neg_ids].max().item()
            results[id] = {
                "verdict": verdict,
                "log_prob": (pos_log_prob, neg_log_prob),
            }
        except Exception as e:
            traceback.print_exc()

        # print(sequences)
        # print(pos_ids)
        # print(neg_ids)
        # print(response)

    return results

def compute_spatial_distance(data_dir, tokenizer, coco_annotations_path, universal_id_prefix, layer, postfix=None):
    coco = load_coco_annotations(coco_annotations_path)
    universal_id = torch.load(f"{universal_id_prefix}.pt")
    x_axis = torch.load(f"{universal_id_prefix}_x.pt")[layer]["universal"].to(torch.float32).numpy()
    y_axis = torch.load(f"{universal_id_prefix}_y.pt")[layer]["universal"].to(torch.float32).numpy()

    distances = {}
    for fname in os.listdir(data_dir):
        object_a, object_b, pos_word, neg_word, id = fname.split("_")
        object_a, object_b = object_a.replace("+", " "), object_b.replace("+", " ")
        id = int(id)
        token_a, token_b = [
            tokenizer.tokenize(" " + tw)[-1] for tw in [object_a, object_b]
        ]
        
        x_loc = []
        y_loc = []
        gt_x_loc = []
        gt_y_loc = []
        for object, token, color in [(object_a, token_a, "red"), (object_b, token_b, "blue")]:
            #print(universal_id[LAYER])
            _, gt_embeds = get_subject_spatial_id(id, object, coco, universal_id[layer]["universal"])
            gt_embeds = gt_embeds.to(torch.float32).numpy()[np.newaxis,:]
            text_embeds = torch.load(path.join(data_dir, fname, "text.pt"))[layer][token].to(torch.float32).numpy()[np.newaxis,:]
            if postfix is None:
                embeds = torch.load(path.join(data_dir, fname, "embeds.pt"))[layer][token].to(torch.float32).numpy() - text_embeds
            else:
                embeds = torch.load(path.join(data_dir, fname, f"embeds_{postfix}.pt"))[layer][token].to(torch.float32).numpy() - text_embeds

            gt_coords, _ = project_onto_plane(gt_embeds, x_axis, y_axis)
            coords, _ = project_onto_plane(embeds, x_axis, y_axis)
            #dis = euclidean_distances(coords, gt_coords)
            x_loc.append(coords[0, 0])
            gt_x_loc.append(gt_coords[0, 0])
            y_loc.append(coords[0, 1])
            gt_y_loc.append(gt_coords[0, 1])
        if pos_word == "right":
            distances[id] = ((x_loc[0] - x_loc[1]).item() - (gt_x_loc[0] - gt_x_loc[1]).item())
        elif pos_word == "left":
            distances[id] = ((x_loc[1] - x_loc[0]).item() - (gt_x_loc[1] - gt_x_loc[0]).item())
        elif pos_word == "below":
            distances[id] = ((y_loc[0] - y_loc[1]).item() - (gt_y_loc[0] - gt_y_loc[1]).item())
        elif pos_word == "above":
            distances[id] = ((y_loc[1] - y_loc[0]).item() - (gt_y_loc[1] - gt_y_loc[0]).item())

    return distances

def accuracy_vs_steerability(data_dir, model_name):
    model_str = VLMForExtraction.get_model_string(model_name)
    tokenizer = AutoProcessor.from_pretrained(model_str).tokenizer
    test_dir = os.listdir(data_dir)[0]
    postfix = set()
    files = os.listdir(path.join(data_dir, test_dir))
    for fname in files:
        if "sequences_" in fname:
            postfix.add(fname.split("sequences_")[-1].split(".")[0])

    results = {}
    results["none"] = compute_verdict(data_dir, tokenizer, model_name)
    for p in postfix:
        results[p] = compute_verdict(data_dir, tokenizer, model_name, p)
    return results


def easy_vs_hard_split(data_dir, model, coco_annotation, universal_id, layer):
    model = VLMForExtraction(model)
    tokenizer = model.processor.tokenizer
    distances = compute_spatial_distance(
        data_dir, tokenizer, coco_annotation, universal_id, layer
    )
    # sort by value
    distances = dict(sorted(distances.items(), key=lambda item: item[1]))

    items = list(distances.items())  # list of (key, value) pairs
    mid = len(items) // 2

    hard = [k for k, v in items[:mid]]  # first half keys
    easy = [k for k, v in items[mid:]]  # second half keys

    return easy, hard


if __name__ == "__main__":
    # easy, hard = easy_vs_hard_split(
    #     "embeds/coco_train_big/qwen2-2b",
    #     "qwen2-2b",
    #     "images/coco/annotations/instances_train2017.json",
    #     "embeds/universal_id/qwen2-2b",
    #     12,
    # )
    # with open("coco_easy.json", "w") as f:
    #     json.dump(easy, f)

    # with open("coco_hard.json", "w") as f:
    #     json.dump(hard, f)
    parser = argparse.ArgumentParser(description="Compute COCO Stats")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing data",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Output path",
    )
    args = parser.parse_args()
    results = accuracy_vs_steerability(args.data_dir, args.model)

    with open(args.output_path, "w") as f:
        json.dump(results, f)
