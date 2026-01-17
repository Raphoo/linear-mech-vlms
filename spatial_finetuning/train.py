# qwen_finetune_spatial.py
# Fine-tune Qwen2-VL for spatial reasoning with custom image dataset
# Training: Custom spatial reasoning images with format "object name 1_x-y_object name 2_x-y.png"
# Validation: COCO dataset
# Generates spatial reasoning questions: left/right and above/below

import argparse, json, os, re, math, random, time
from typing import List, Tuple, Dict, Any, Optional, Iterable, Set

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoProcessor,
    AutoConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    LlavaForConditionalGeneration,
    MllamaForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, metrics will not be logged to W&B")

# ---------------------------
# Prompt & caption parsing
# ---------------------------
REL_PATTERNS = [
    re.compile(
        r"^.*?\bphoto of\b\s*(.+?)\s+to the\s+(left|right)\s+of\s+(.+?)[\.\s]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^.*?\bphoto of\b\s*(.+?)\s+(above|below)\s+(?:the\s+|a\s+)?(.+?)[\.\s]*$",
        re.IGNORECASE,
    ),
]


def parse_relation_caption(caption: str) -> Tuple[str, str, str]:
    cap = caption.strip()
    for pat in REL_PATTERNS:
        m = pat.match(cap)
        if m:
            x, rel, y = (
                m.group(1).strip(),
                m.group(2).lower().strip(),
                m.group(3).strip(),
            )
            return x, rel, y
    raise ValueError(f"Unrecognized caption pattern: {caption}")


def question_from_relation(x: str, rel: str, y: str) -> Tuple[str, List[str]]:
    if rel in ("left", "right"):
        return (
            f'Is {x} to the left or right of {y}? Answer with "left" or "right" only.',
            [
                "left",
                "right",
            ],
        )
    elif rel in ("above", "below"):
        return f'Is {x} above or below {y}? Answer with "above" or "below" only.', [
            "above",
            "below",
        ]
    else:
        raise ValueError(rel)


def normalize_prediction(text: str, opts: List[str]) -> str:
    t = text.strip().lower()
    for opt in opts:
        if re.search(rf"\b{opt}\b", t):
            return opt
    for opt in opts:
        if t.startswith(opt):
            return opt
    return ""


# ---------------------------
# COCO Dataset (for validation)
# ---------------------------
def coco_id_to_path(image_id: int, root_dir: str) -> str:
    fname = f"{int(image_id):012d}.jpg"
    return os.path.join(root_dir, fname)


class COCOPairsDataset(Dataset):
    def __init__(self, json_path: str, image_root: str):
        self.image_root = image_root
        with open(json_path, "r") as f:
            raw = json.load(f)
        self.items = []
        for row in raw:
            if not isinstance(row, list) or len(row) < 3:
                continue
            img_id, cap_pos, cap_neg = row[0], row[1], row[2]
            try:
                x, rel, y = parse_relation_caption(cap_pos)
            except Exception:
                x, rel, y = parse_relation_caption(cap_neg)
            answer = rel.lower()
            question, opts = question_from_relation(x, rel, y)
            image_path = coco_id_to_path(img_id, image_root)
            self.items.append(
                {
                    "image_path": image_path,
                    "question": question,
                    "answer": answer,
                    "opts": opts,
                }
            )
        if len(self.items) == 0:
            raise RuntimeError(f"No valid samples parsed from {json_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


# ---------------------------
# Dataset for Spatial Reasoning (for training)
# ---------------------------
class SpatialReasoningDataset(Dataset):
    def __init__(self, image_dir: str, processor=None):
        """
        Dataset for spatial reasoning from images with filename format:
        "object name 1_x-y_object name 2_x-y.png"

        Args:
            image_dir: Directory containing the images
            processor: Qwen2VLProcessor for tokenization (optional, for spatial ID)
        """
        self.image_dir = image_dir
        self.processor = processor
        self.items = []
        self.object_pair_to_indices = {}  # Maps (obj1, obj2) -> list of indices
        self.fixed_obj1_groups = (
            {}
        )  # Maps (obj1, x1, y1, obj2) -> list of indices for spatial ID

        # List all PNG files in the directory
        if not os.path.exists(image_dir):
            raise RuntimeError(f"Image directory not found: {image_dir}")

        image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

        for img_file in image_files:
            # Parse filename to extract objects and coordinates
            parsed = self.parse_filename(img_file)
            if parsed is None:
                print(f"Warning: Could not parse filename: {img_file}")
                continue

            obj1, x1, y1, obj2, x2, y2 = parsed
            image_path = os.path.join(image_dir, img_file)

            # Tokenize object names to get target tokens (last token with leading space)
            target_tokens = {}
            if processor is not None:
                for obj_name in [obj1, obj2]:
                    tokens = processor.tokenizer.tokenize(" " + obj_name)
                    if tokens:
                        target_tokens[obj_name] = tokens[-1]
                    else:
                        target_tokens[obj_name] = None

            # Generate left/right question if x coordinates differ
            if x1 != x2:
                if x1 < x2:
                    answer = "left"
                else:
                    answer = "right"
                question = f'Is {obj1} to the left or right of {obj2}? Answer with "left" or "right" only.'
                item_idx = len(self.items)
                self.items.append(
                    {
                        "image_path": image_path,
                        "question": question,
                        "answer": answer,
                        "opts": ["left", "right"],
                        "obj1": obj1,
                        "obj2": obj2,
                        "coords": (x1, y1, x2, y2),
                        "target_tokens": target_tokens,
                    }
                )
                # Track object pair grouping
                pair_key = (obj1, obj2)
                if pair_key not in self.object_pair_to_indices:
                    self.object_pair_to_indices[pair_key] = []
                self.object_pair_to_indices[pair_key].append(item_idx)

                # Track fixed obj1 position grouping for spatial ID
                fixed_key = (obj1, x1, y1, obj2)
                if fixed_key not in self.fixed_obj1_groups:
                    self.fixed_obj1_groups[fixed_key] = []
                self.fixed_obj1_groups[fixed_key].append(item_idx)

            # Generate above/below question if y coordinates differ
            if y1 != y2:
                # In image coordinates, smaller y means higher position (above)
                if y1 < y2:
                    answer = "above"
                else:
                    answer = "below"
                question = f'Is {obj1} above or below {obj2}? Answer with "above" or "below" only.'
                item_idx = len(self.items)
                self.items.append(
                    {
                        "image_path": image_path,
                        "question": question,
                        "answer": answer,
                        "opts": ["above", "below"],
                        "obj1": obj1,
                        "obj2": obj2,
                        "coords": (x1, y1, x2, y2),
                        "target_tokens": target_tokens,
                    }
                )
                # Track object pair grouping
                pair_key = (obj1, obj2)
                if pair_key not in self.object_pair_to_indices:
                    self.object_pair_to_indices[pair_key] = []
                self.object_pair_to_indices[pair_key].append(item_idx)

                # Track fixed obj1 position grouping for spatial ID
                fixed_key = (obj1, x1, y1, obj2)
                if fixed_key not in self.fixed_obj1_groups:
                    self.fixed_obj1_groups[fixed_key] = []
                self.fixed_obj1_groups[fixed_key].append(item_idx)

        if len(self.items) == 0:
            raise RuntimeError(f"No valid samples found in {image_dir}")

        print(
            f"Loaded {len(self.items)} question-answer pairs from {len(image_files)} images"
        )
        print(f"Found {len(self.object_pair_to_indices)} unique object pairs")
        print(
            f"Found {len(self.fixed_obj1_groups)} unique fixed-obj1 groups for spatial ID"
        )

    def parse_filename(self, filename: str):
        """
        Parse filename format: "object name 1_x-y_object name 2_x-y.png"
        Returns: (obj1, x1, y1, obj2, x2, y2) or None if parsing fails
        """
        # Remove .png extension
        if not filename.endswith(".png"):
            return None
        name = filename[:-4]

        # Find all coordinate patterns _digits-digits
        coord_pattern = re.compile(r"_(\d+)-(\d+)")
        matches = list(coord_pattern.finditer(name))

        if len(matches) < 2:
            return None

        # Get the last two matches (coordinates for obj1 and obj2)
        match1 = matches[-2]  # object 1's coordinates
        match2 = matches[-1]  # object 2's coordinates

        # Extract coordinates
        x1, y1 = int(match1.group(1)), int(match1.group(2))
        x2, y2 = int(match2.group(1)), int(match2.group(2))

        # Extract object names (everything before the coordinate pattern)
        obj1 = name[: match1.start()].strip()
        obj2 = name[match1.end() : match2.start()].strip()

        # Remove leading underscore from obj2 if present
        if obj2.startswith("_"):
            obj2 = obj2[1:]

        return obj1, x1, y1, obj2, x2, y2

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]

    def get_object_pairs(self):
        """Returns list of object pair keys"""
        return list(self.object_pair_to_indices.keys())

    def get_fixed_obj1_groups(self):
        """Returns list of fixed obj1 position group keys"""
        return list(self.fixed_obj1_groups.keys())


def split_object_pairs(
    dataset: SpatialReasoningDataset, val_frac: float, seed: int
) -> Dict[str, Any]:
    """Split dataset indices by holding out entire object pairs for validation."""

    pair_keys = dataset.get_object_pairs()
    num_pairs = len(pair_keys)
    if val_frac <= 0 or num_pairs < 2:
        return {
            "train_indices": list(range(len(dataset))),
            "holdout_indices": [],
            "train_pairs": set(pair_keys),
            "holdout_pairs": set(),
        }

    rng = random.Random(seed)
    shuffled_pairs = pair_keys[:]
    rng.shuffle(shuffled_pairs)

    holdout_pair_count = max(1, int(num_pairs * val_frac))
    if holdout_pair_count >= num_pairs:
        holdout_pair_count = num_pairs - 1
    if holdout_pair_count <= 0:
        return {
            "train_indices": list(range(len(dataset))),
            "holdout_indices": [],
            "train_pairs": set(pair_keys),
            "holdout_pairs": set(),
        }

    holdout_pairs = set(shuffled_pairs[:holdout_pair_count])
    train_indices: List[int] = []
    holdout_indices: List[int] = []
    train_pairs: Set[Tuple[str, str]] = set()

    for pair_key, indices in dataset.object_pair_to_indices.items():
        if pair_key in holdout_pairs:
            holdout_indices.extend(indices)
        else:
            train_indices.extend(indices)
            train_pairs.add(pair_key)

    return {
        "train_indices": sorted(train_indices),
        "holdout_indices": sorted(holdout_indices),
        "train_pairs": train_pairs,
        "holdout_pairs": holdout_pairs,
    }


# ---------------------------
# Custom Batch Sampler for Spatial ID Training
# ---------------------------
class FixedObj1BatchSampler:
    """
    Samples batches where obj1 is at a fixed position and obj2 varies across the grid.
    Each batch contains samples with the same (obj1, x1, y1, obj2) where obj2 is at different positions.
    Can split large groups into smaller sub-batches to fit in memory.
    """

    def __init__(
        self,
        dataset: SpatialReasoningDataset,
        max_batch_size: int = 8,
        shuffle: bool = True,
        allowed_indices: Optional[Iterable[int]] = None,
    ):
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.allowed = set(allowed_indices) if allowed_indices is not None else None

        self.group_indices: List[List[int]] = []
        for group_key in dataset.get_fixed_obj1_groups():
            indices = dataset.fixed_obj1_groups[group_key]
            if self.allowed is not None:
                indices = [idx for idx in indices if idx in self.allowed]
            if len(indices) > 1:
                self.group_indices.append(indices)

    def __iter__(self):
        groups = [grp.copy() for grp in self.group_indices]
        if self.shuffle:
            random.shuffle(groups)

        for indices in groups:
            if self.shuffle:
                random.shuffle(indices)

            for i in range(0, len(indices), self.max_batch_size):
                batch = indices[i : i + self.max_batch_size]
                if len(batch) > 1:
                    yield batch

    def __len__(self):
        total_batches = 0
        for indices in self.group_indices:
            group_size = len(indices)
            if group_size > 1:
                total_batches += (
                    group_size + self.max_batch_size - 1
                ) // self.max_batch_size
        return total_batches


# ---------------------------
# Collator (model-agnostic)
# ---------------------------
class VLMChatCollator:
    def __init__(self, processor, max_length: int = 6144):
        """
        Model-agnostic collator for VLMs.
        Works with Qwen2VL, LLaVA, and LLaMA processors.
        """
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        conv_full, conv_prefix = [], []
        for ex in batch:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image", "path": ex["image_path"]},
                    {"type": "text", "text": ex["question"]},
                ],
            }
            conv_full.append(
                [
                    user_msg,
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": ex["answer"]}],
                    },
                ]
            )
            conv_prefix.append([user_msg])

        full_inputs = self.processor.apply_chat_template(
            conv_full,
            tokenize=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_dict=True,
        )
        prefix_inputs = self.processor.apply_chat_template(
            conv_prefix,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_dict=True,
        )

        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        labels = input_ids.clone()

        prefix_am = prefix_inputs["attention_mask"]
        for i in range(input_ids.size(0)):
            prefix_len = int(prefix_am[i].sum().item())
            labels[i, :prefix_len] = -100

        has_target = labels.ne(-100).any(dim=1)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "has_target": has_target,
            "batch_data": batch,  # Include original batch data for spatial ID loss
        }
        # Include all vision-related keys (different models need different keys)
        for k in ["pixel_values", "image_grid_thw", "aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask"]:
            if k in full_inputs:
                model_inputs[k] = full_inputs[k]
        return model_inputs


# ---------------------------
# Budget-aware LM parameter selection
# ---------------------------
# Try to be robust to different internal namings:
# Qwen2-VL typically nests decoder under "language_model.model.layers.<idx>."
LM_BLOCK_RE = re.compile(
    r"(?:^|\.)(?:language_model\.)?(?:model\.)?(?:decoder\.)?(?:layers?)\.(\d+)\."
)
VISION_BLOCK_RE = re.compile(r"\.(?:blocks|layers)\.(\d+)\.")


def total_num_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def is_vision(n: str) -> bool:
    return ("vision" in n) or ("visual" in n) or ("vit" in n)


def is_lm(n: str) -> bool:
    # Include the decoder stack and lm_head; exclude anything vision-ish.
    if is_vision(n):
        return False
    return (
        "language_model" in n
        or ".model.layers." in n
        or ".decoder.layers." in n
        or n.startswith("model.layers.")
        or "lm_head" in n
    )


def block_index_lm(n: str) -> int:
    m = LM_BLOCK_RE.search(n)
    return int(m.group(1)) if m else -1


def block_index_vision(n: str) -> int:
    m = VISION_BLOCK_RE.search(n)
    return int(m.group(1)) if m else -1


def looks_like_mlp_weight(n: str) -> bool:
    # LLaMA/Qwen2-style MLP: up_proj, gate_proj, down_proj; also generic fc1/fc2/intermediate
    return any(
        k in n
        for k in [
            "mlp.",
            "up_proj",
            "down_proj",
            "gate_proj",
            "intermediate",
            "fc1",
            "fc2",
            ".ffn",
        ]
    ) and n.endswith(".weight")


def looks_like_attn_weight(n: str) -> bool:
    # q_proj, k_proj, v_proj, o_proj, attn/attention
    return (
        "attn" in n
        or "attention" in n
        or any(x in n for x in ["q_proj", "k_proj", "v_proj", "o_proj"])
    ) and n.endswith(".weight")


def looks_like_norm(n: str) -> bool:
    return (
        (".ln" in n)
        or ("layernorm" in n)
        or (".norm" in n)
        or (".input_layernorm" in n)
        or (".post_attention_layernorm" in n)
    )


def looks_like_projector(n: str) -> bool:
    return any(
        k in n
        for k in [
            "mm_projector",
            "multi_modal_projector",
            "vision_proj",
            "visual_projector",
            "image_projector",
            "resampler",
        ]
    )


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def get_model_string(model_alias: str) -> str:
    """Map model alias to HuggingFace model string."""
    mapping = {
        "llava-7b": "llava-hf/llava-1.5-7b-hf",
        "qwen2-2b": "Qwen/Qwen2-VL-2B-Instruct",
        "llama-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    }
    if model_alias in mapping:
        return mapping[model_alias]
    # If not an alias, assume it's a direct model string
    return model_alias


def load_model_and_processor(model_name: str, device, bf16: bool = False):
    """
    Load model and processor based on model name/alias.
    Returns: (model, processor, model_type)
    model_type is one of: "qwen2vl", "llava", "llama"
    """
    model_str = get_model_string(model_name)

    # Determine model type based on model string
    if "qwen" in model_str.lower():
        model_type = "qwen2vl"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_str,
            torch_dtype=torch.bfloat16 if bf16 else None,
            device_map=None,
        )
        processor = Qwen2VLProcessor.from_pretrained(model_str)
    elif "llava" in model_str.lower():
        model_type = "llava"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_str,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            device_map=None,
        )
        processor = AutoProcessor.from_pretrained(model_str)
    elif "llama" in model_str.lower():
        model_type = "llama"
        model = MllamaForConditionalGeneration.from_pretrained(
            model_str,
            torch_dtype=torch.bfloat16,
            device_map=None,
        )
        processor = AutoProcessor.from_pretrained(model_str)
    else:
        raise ValueError(f"Unsupported model: {model_name} (resolved to {model_str})")

    model = model.to(device)
    return model, processor, model_type


def greedy_select(
    trainables: List[Tuple[str, torch.nn.Parameter, int]],
    budget: int,
    bitfit_bias_only: bool,
):
    chosen, used = [], 0
    trainables = sorted(trainables, key=lambda x: x[2], reverse=True)
    for n, p, sz in trainables:
        if bitfit_bias_only and not n.endswith(".bias"):
            continue
        if used + sz <= budget:
            p.requires_grad = True
            chosen.append((n, sz))
            used += sz
        if used >= budget:
            break
    return chosen, used


def apply_budget_selection_lm(
    model,
    selector: str,
    budget_frac: float,
    last_k_blocks: int,
    bitfit_bias_only: bool,
    include_lm_head: bool = True,
):
    """
    selector ∈ {"lm_mlp_lastk","lm_attn_lastk","lm_norms_lastk","random_lm","lm_head_only"}
    """
    tparams = total_num_params(model)
    budget = max(1, int(tparams * budget_frac))

    candidates: List[Tuple[str, torch.nn.Parameter, int]] = []

    # Optionally include lm_head (small but can help with calibration)
    if include_lm_head or selector == "lm_head_only":
        for n, p in model.named_parameters():
            if "lm_head" in n:
                candidates.append((n, p, p.numel()))

    if selector != "lm_head_only":
        all_lm = [(n, p, p.numel()) for (n, p) in model.named_parameters() if is_lm(n)]
        if selector == "random_lm":
            random.shuffle(all_lm)
            pool = all_lm
        else:
            if selector == "lm_mlp_lastk":
                pool = [
                    x
                    for x in all_lm
                    if looks_like_mlp_weight(x[0])
                    or (bitfit_bias_only and x[0].endswith(".bias"))
                ]
            elif selector == "lm_attn_lastk":
                pool = [
                    x
                    for x in all_lm
                    if looks_like_attn_weight(x[0])
                    or (bitfit_bias_only and x[0].endswith(".bias"))
                ]
            elif selector == "lm_norms_lastk":
                pool = [x for x in all_lm if looks_like_norm(x[0])]
            else:
                pool = all_lm

            pool.sort(key=lambda x: block_index_lm(x[0]), reverse=True)
            if last_k_blocks > 0:
                pool = [x for x in pool if (block_index_lm(x[0]) >= 0)]
                max_b = max([block_index_lm(x[0]) for x in pool]) if pool else -1
                keep_min = max(0, max_b - last_k_blocks + 1)
                pool = [x for x in pool if block_index_lm(x[0]) >= keep_min]
        candidates.extend(pool)

    chosen, used = greedy_select(candidates, budget, bitfit_bias_only=bitfit_bias_only)
    trainable_params = sum(sz for _, sz in chosen)
    pct = 100.0 * trainable_params / tparams
    return {
        "total_params": tparams,
        "budget": budget,
        "used": trainable_params,
        "pct": pct,
        "chosen": [n for n, _ in chosen],
    }


def apply_budget_selection_vision(
    model,
    selector: str,
    budget_frac: float,
    last_k_blocks: int,
    bitfit_bias_only: bool,
    also_projector: bool = True,
):
    """Budgeted selection over the vision encoder stack."""

    tparams = total_num_params(model)
    budget = max(1, int(tparams * budget_frac))

    candidates: List[Tuple[str, torch.nn.Parameter, int]] = []

    if also_projector or selector == "projector_only":
        for n, p in model.named_parameters():
            if looks_like_projector(n):
                candidates.append((n, p, p.numel()))

    if selector in {
        "vision_mlp_lastk",
        "vision_attn_lastk",
        "vision_norms_lastk",
        "projector_only",
        "random",
    }:
        all_vis = [
            (n, p, p.numel()) for (n, p) in model.named_parameters() if is_vision(n)
        ]

        if selector == "random":
            random.shuffle(all_vis)
            candidates.extend(all_vis)
        elif selector != "projector_only":
            if selector == "vision_mlp_lastk":
                pool = [
                    x
                    for x in all_vis
                    if looks_like_mlp_weight(x[0])
                    or (bitfit_bias_only and x[0].endswith(".bias"))
                ]
            elif selector == "vision_attn_lastk":
                pool = [
                    x
                    for x in all_vis
                    if looks_like_attn_weight(x[0])
                    or (bitfit_bias_only and x[0].endswith(".bias"))
                ]
            elif selector == "vision_norms_lastk":
                pool = [x for x in all_vis if looks_like_norm(x[0])]
            else:
                pool = all_vis

            pool.sort(key=lambda x: block_index_vision(x[0]), reverse=True)
            if last_k_blocks > 0:
                pool = [x for x in pool if block_index_vision(x[0]) >= 0]
                if pool:
                    max_b = max(block_index_vision(x[0]) for x in pool)
                    keep_min = max(0, max_b - last_k_blocks + 1)
                    pool = [x for x in pool if block_index_vision(x[0]) >= keep_min]
            candidates.extend(pool)

    chosen, used = greedy_select(candidates, budget, bitfit_bias_only=bitfit_bias_only)
    trainable_params = sum(sz for _, sz in chosen)
    pct = 100.0 * trainable_params / tparams
    return {
        "total_params": tparams,
        "budget": budget,
        "used": trainable_params,
        "pct": pct,
        "chosen": [n for n, _ in chosen],
    }


# ---------------------------
# Spatial ID Loss Computation
# ---------------------------
def compute_spatial_id_loss(
    batch_data: List[Dict],
    hidden_states,
    input_ids,
    processor,
    universal_id_dict,
    layer_idx: int,
    image_size: int = 224,
):
    """
    Compute spatial ID loss for a batch with the same object pair.

    Args:
        batch_data: List of data items from the batch (with obj1, obj2, coords, target_tokens)
        hidden_states: Hidden states from model output (tuple of layers)
        input_ids: Input token IDs (batch_size, seq_len)
        processor: Qwen2VLProcessor for tokenization
        universal_id_dict: Pre-computed universal spatial IDs {layer: {"universal": {(x,y,size): embedding}}}
        layer_idx: Which layer to extract embeddings from
        image_size: Image size for spatial ID lookup (default: 224)

    Returns:
        Spatial ID loss (cosine similarity based)
    """
    if layer_idx not in universal_id_dict:
        return torch.tensor(0.0, device=input_ids.device)

    if "universal" not in universal_id_dict[layer_idx]:
        return torch.tensor(0.0, device=input_ids.device)

    universal_ids = universal_id_dict[layer_idx]["universal"]

    # Get hidden states for the specified layer
    layer_hidden = hidden_states[layer_idx]  # (batch_size, seq_len, hidden_dim)

    # Extract target token for this batch (all should have same objects)
    first_item = batch_data[0]
    target_token = first_item["target_tokens"].get(first_item["obj2"])

    if target_token is None:
        return torch.tensor(0.0, device=input_ids.device)

    # Convert input_ids to tokens
    batch_size = input_ids.shape[0]
    device = input_ids.device

    # Extract embeddings for each sample in the batch
    embeddings = []
    coords_list = []

    for i in range(batch_size):
        tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[i].tolist())

        # Find target token positions (we want the last occurrence)
        token_positions = [j for j, tok in enumerate(tokens) if tok == target_token]

        if not token_positions:
            continue

        # Get the embedding at the last occurrence of the target token
        pos = token_positions[-1]

        # Bounds check to prevent illegal memory access
        if pos >= layer_hidden.shape[1]:
            continue

        embedding = layer_hidden[i, pos, :]  # (hidden_dim,)
        embeddings.append(embedding)

        # Get ground-truth coordinates
        x1, y1, x2, y2 = batch_data[i]["coords"]
        coords_list.append((x2, y2, image_size))  # Use obj2 coordinates

    if len(embeddings) < 2:
        # Need at least 2 samples to compute spatial ID
        return torch.tensor(0.0, device=device)

    # Stack embeddings
    embeddings = torch.stack(embeddings)  # (num_samples, hidden_dim)

    # Compute mean-centered embeddings (spatial IDs)
    mean_embed = embeddings.mean(dim=0, keepdim=True)  # (1, hidden_dim)
    spatial_ids = embeddings - mean_embed  # (num_samples, hidden_dim)

    # Get ground-truth spatial IDs from universal_id_dict
    gt_spatial_ids = []
    valid_indices = []

    for idx, coords in enumerate(coords_list):
        if coords in universal_ids:
            gt_spatial_ids.append(universal_ids[coords].to(device))
            valid_indices.append(idx)

    if len(gt_spatial_ids) == 0:
        return torch.tensor(0.0, device=device)

    gt_spatial_ids = torch.stack(gt_spatial_ids)  # (num_valid, hidden_dim)
    valid_spatial_ids = spatial_ids[valid_indices]  # (num_valid, hidden_dim)

    # Compute cosine similarity loss (we want to maximize similarity, so minimize 1 - cos_sim)
    cos_sim = torch.nn.functional.cosine_similarity(
        valid_spatial_ids, gt_spatial_ids, dim=1
    )  # (num_valid,)

    # Loss is 1 - cosine similarity (ranges from 0 to 2, minimize to maximize similarity)
    # cos_sim = 1.0 → loss = 0.0 (perfect alignment)
    # cos_sim = 0.0 → loss = 1.0 (orthogonal)
    # cos_sim = -1.0 → loss = 2.0 (opposite direction)
    loss = 1.0 - cos_sim.mean()

    return loss


# ---------------------------
# Evaluation
# ---------------------------
@torch.no_grad()
def evaluate(
    model,
    processor,
    dataset: Dataset,
    device: torch.device,
    max_samples: int = None,
) -> float:
    model.eval()
    idxs = list(range(len(dataset)))
    if max_samples:
        idxs = idxs[:max_samples]
    correct = 0
    for i in idxs:
        ex = dataset[i]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": ex["image_path"]},
                    {"type": "text", "text": ex["question"]},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen = model.generate(**inputs, max_new_tokens=5)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], gen)]
        pred_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        pred = normalize_prediction(pred_text, ex["opts"])

        if pred == ex["answer"]:
            correct += 1
    return correct / max(1, len(idxs))


# ---------------------------
# Training loop (8-bit Adam optional; checkpointing optional)
# ---------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb and generate run name
    use_wandb = WANDB_AVAILABLE and not args.no_wandb

    # Generate run name based on key hyperparameters
    run_name_parts = []
    run_name_parts.append(args.model_name)  # model name/alias
    run_name_parts.append(args.selector)  # selector includes lm/vision prefix
    run_name_parts.append(f"b{args.param_budget_frac}")  # budget fraction
    run_name_parts.append(f"bs{args.batch_size}")  # batch size
    run_name_parts.append(f"lr{args.lr:.0e}")  # learning rate in scientific notation
    if args.universal_id_path:
        run_name_parts.append(f"sid{args.spatial_id_layer}")  # spatial id layer
        run_name_parts.append(f"sw{args.spatial_id_weight}")  # spatial id weight
    run_name = "_".join(run_name_parts)

    # Use run name as output directory
    output_dir = f"./runs/{run_name}"

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )

    # Load model and processor based on model alias
    model, processor, model_type = load_model_and_processor(
        args.model_name, device, bf16=args.bf16
    )
    model.config.use_cache = False
    print(f"Loaded model type: {model_type}")

    freeze_all(model)
    if args.selector.startswith("vision"):
        sel_info = apply_budget_selection_vision(
            model,
            selector=args.selector,
            budget_frac=args.param_budget_frac,
            last_k_blocks=args.last_k_blocks,
            bitfit_bias_only=args.bitfit_bias_only,
            also_projector=not args.no_projector,
        )
        sel_label = "[Budget-Vision]"
    else:
        sel_info = apply_budget_selection_lm(
            model,
            selector=args.selector,
            budget_frac=args.param_budget_frac,
            last_k_blocks=args.last_k_blocks,
            bitfit_bias_only=args.bitfit_bias_only,
            include_lm_head=not args.no_lm_head,
        )
        sel_label = "[Budget-LM]"

    print(
        f"{sel_label} total={sel_info['total_params']:,}  budget={sel_info['budget']:,}  "
        f"used={sel_info['used']:,} ({sel_info['pct']:.3f}%)"
    )
    print(f"[Trainable tensors] {len(sel_info['chosen'])} modules (showing first 20):")
    for n in sel_info["chosen"][:20]:
        print("  -", n)

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()

    # Load universal spatial ID dict if provided
    universal_id_dict = None
    if args.universal_id_path:
        print(f"Loading universal spatial IDs from {args.universal_id_path}")
        universal_id_dict = torch.load(
            args.universal_id_path, map_location="cpu", weights_only=False
        )
        print(
            f"Spatial ID loss enabled: layer={args.spatial_id_layer}, weight={args.spatial_id_weight}"
        )

    # Data
    spatial_dataset = SpatialReasoningDataset(args.train_dir, processor=processor)
    val_ds = COCOPairsDataset(args.val_json, args.coco_val_dir)
    collator = VLMChatCollator(processor, max_length=4096)

    split_info = split_object_pairs(
        spatial_dataset,
        val_frac=args.spatial_holdout_frac,
        seed=args.spatial_holdout_seed,
    )
    train_indices = split_info["train_indices"]
    holdout_indices = split_info["holdout_indices"]
    if len(train_indices) == 0:
        raise RuntimeError("No spatial training samples remain after holdout split.")

    train_subset = Subset(spatial_dataset, train_indices)
    holdout_ds = Subset(spatial_dataset, holdout_indices) if holdout_indices else None

    print(
        "Spatial split: "
        f"{len(train_indices)} train samples across {len(split_info['train_pairs'])} pairs | "
        f"{len(holdout_indices)} holdout samples across {len(split_info['holdout_pairs'])} pairs"
    )

    os.makedirs(output_dir, exist_ok=True)

    def run_eval_and_checkpoint(step: int):
        print(f"Running evaluation/checkpoint at step {step}...")
        model.eval()
        val_at_step = evaluate(
            model,
            processor,
            val_ds,
            device,
            max_samples=args.eval_max_samples,
        )
        print(f"Validation accuracy (step {step}): {val_at_step:.3f}")
        if use_wandb:
            wandb.log({"val/accuracy": val_at_step}, step=step)

        if holdout_ds is not None and len(holdout_ds) > 0:
            holdout_at_step = evaluate(
                model,
                processor,
                holdout_ds,
                device,
                max_samples=args.eval_max_samples,
            )
            eval_sample_count = (
                args.eval_max_samples if args.eval_max_samples else len(holdout_ds)
            )
            print(
                f"Spatial holdout accuracy (step {step}): {holdout_at_step:.3f} "
                f"on {eval_sample_count} samples"
            )
            if use_wandb:
                wandb.log({"holdout/accuracy": holdout_at_step}, step=step)
        ckpt_dir = os.path.join(output_dir, f"checkpoint-step{step:06d}")
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        print(f"Saved checkpoint to {ckpt_dir}")
        model.train()

    # Use custom batch sampler if spatial ID is enabled
    if universal_id_dict is not None:
        # Use same batch size as baseline for fair comparison
        max_batch = args.batch_size
        batch_sampler = FixedObj1BatchSampler(
            spatial_dataset,
            max_batch_size=max_batch,
            shuffle=True,
            allowed_indices=train_indices,
        )
        train_loader = DataLoader(
            spatial_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=8,
            pin_memory=True,
        )
        print(
            f"Using FixedObj1BatchSampler: {len(batch_sampler)} batches of max {max_batch} samples each"
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=8,
            pin_memory=True,
        )

    # Optimizer (8-bit optional)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.adam8bit:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                trainable_params, lr=args.lr, weight_decay=0.01
            )
        except Exception as e:
            print("Falling back to torch AdamW (bitsandbytes not available):", e)
            optimizer = torch.optim.AdamW(
                trainable_params, lr=args.lr, weight_decay=0.01
            )
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = math.ceil(len(train_loader) * args.epochs / max(1, args.grad_accum))
    warmup_steps = int(0.03 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Pre eval
    print("Evaluating BEFORE fine-tuning...")
    base_acc = evaluate(
        model, processor, val_ds, device, max_samples=args.eval_max_samples
    )
    print(f"Validation accuracy (pre): {base_acc:.3f}")
    if use_wandb:
        wandb.log({"val/accuracy_pre": base_acc}, step=0)

    holdout_acc_pre = None
    if holdout_ds is not None and len(holdout_ds) > 0:
        holdout_acc_pre = evaluate(
            model,
            processor,
            holdout_ds,
            device,
            max_samples=args.eval_max_samples,
        )
        print(
            f"Spatial holdout accuracy (pre): {holdout_acc_pre:.3f} on {len(holdout_ds)} samples"
        )
        if use_wandb:
            wandb.log({"holdout/accuracy_pre": holdout_acc_pre}, step=0)

    model.train()

    # Verify trainable parameters before starting
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Starting training with {trainable_count} trainable parameters")
    print(f"First 5 trainable parameters:")
    for i, (n, p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            print(f"  {n}: {p.shape}")
            if i >= 4:
                break

    global_step = 0
    print("Start training")
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader, start=1):
            has_t = batch.pop("has_target")
            batch_data = batch.pop("batch_data", None)  # Extract original batch data

            if not bool(has_t.any()):
                continue
            keep = has_t.nonzero(as_tuple=False).squeeze(-1)

            # Filter batch data as well
            if batch_data is not None:
                batch_data = [batch_data[i] for i in keep.tolist()]

            batch = {
                k: (
                    v.index_select(0, keep)
                    if v.dim() > 0 and v.size(0) == has_t.size(0)
                    else v
                )
                for k, v in batch.items()
            }
            batch_tensors = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }

            # Forward pass with hidden states if spatial ID is enabled
            if step == 1:
                print(
                    f"First batch: input_ids shape = {batch_tensors['input_ids'].shape}"
                )

            if universal_id_dict is not None:
                outputs = model(**batch_tensors, output_hidden_states=True)
            else:
                outputs = model(**batch_tensors)

            if step == 1:
                print(f"Forward pass completed. Loss = {outputs.loss.item():.4f}")

            loss = outputs.loss
            if not loss.requires_grad:
                print(
                    f"WARNING: loss.requires_grad is False at step {step}. Skipping batch."
                )
                print(f"  Batch size: {batch_tensors['input_ids'].shape[0]}")
                print(f"  Loss value: {loss.item():.4f}")
                # Check if any parameters actually have requires_grad
                trainable_count = sum(p.requires_grad for p in model.parameters())
                print(f"  Trainable parameters: {trainable_count}")
                continue

            # Add spatial ID loss if enabled
            spatial_id_loss = None
            if universal_id_dict is not None and batch_data is not None:
                spatial_id_loss = compute_spatial_id_loss(
                    batch_data=batch_data,
                    hidden_states=outputs.hidden_states,
                    input_ids=batch_tensors["input_ids"],
                    processor=processor,
                    universal_id_dict=universal_id_dict,
                    layer_idx=args.spatial_id_layer,
                    image_size=args.spatial_id_image_size,
                )
                # Add spatial ID loss (should be in range [0, 2])
                if (
                    spatial_id_loss.item() < 2.0
                ):  # Only add if valid (filter out edge cases)
                    loss = loss + args.spatial_id_weight * spatial_id_loss

            if step == 1:
                print(
                    f"About to call loss.backward(). loss.requires_grad = {loss.requires_grad}"
                )

            loss.backward()

            if step == 1:
                print(f"Backward pass completed successfully")
            if step % args.grad_accum == 0:
                # Check for invalid gradients before clipping
                valid_grads = True
                try:
                    for p in trainable_params:
                        if p.grad is not None and (
                            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                        ):
                            print(
                                f"WARNING: Invalid gradient detected at step {step}. Skipping update."
                            )
                            valid_grads = False
                            break
                except RuntimeError as e:
                    print(
                        f"WARNING: Error checking gradients at step {step}: {e}. Skipping update."
                    )
                    valid_grads = False

                if valid_grads:
                    try:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        global_step += 1
                    except RuntimeError as e:
                        print(
                            f"WARNING: Error during optimizer step at step {step}: {e}. Skipping update."
                        )

                optimizer.zero_grad(set_to_none=True)
                if global_step % max(1, args.log_every) == 0:
                    if spatial_id_loss is not None:
                        print(
                            f"[epoch {epoch+1}] step {global_step}/{total_steps} "
                            f"loss={loss.item():.4f} (lm={outputs.loss.item():.4f}, "
                            f"spatial_id={spatial_id_loss.item():.4f})"
                        )
                        if use_wandb:
                            wandb.log(
                                {
                                    "train/loss": loss.item(),
                                    "train/lm_loss": outputs.loss.item(),
                                    "train/spatial_id_loss": spatial_id_loss.item(),
                                    "train/epoch": epoch + 1,
                                },
                                step=global_step,
                            )
                    else:
                        print(
                            f"[epoch {epoch+1}] step {global_step}/{total_steps} loss={loss.item():.4f}"
                        )
                        if use_wandb:
                            wandb.log(
                                {
                                    "train/loss": loss.item(),
                                    "train/epoch": epoch + 1,
                                },
                                step=global_step,
                            )

                if args.eval_save_steps > 0 and global_step % args.eval_save_steps == 0:
                    run_eval_and_checkpoint(global_step)

    print("Evaluating AFTER fine-tuning...")
    final_acc = evaluate(
        model, processor, val_ds, device, max_samples=args.eval_max_samples
    )
    print(f"Validation accuracy (post): {final_acc:.3f}")
    if use_wandb:
        wandb.log({"val/accuracy_post": final_acc}, step=global_step)
    if holdout_ds is not None and len(holdout_ds) > 0:
        holdout_acc_post = evaluate(
            model,
            processor,
            holdout_ds,
            device,
            max_samples=args.eval_max_samples,
        )
        print(
            f"Spatial holdout accuracy (post): {holdout_acc_post:.3f} on {len(holdout_ds)} samples"
        )
        if use_wandb:
            wandb.log({"holdout/accuracy_post": holdout_acc_post}, step=global_step)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Directory containing training images (spatial reasoning)",
    )
    p.add_argument(
        "--val_json", type=str, required=True, help="COCO validation JSON file"
    )
    p.add_argument(
        "--coco_val_dir",
        type=str,
        required=True,
        help="COCO validation images directory",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="qwen2-2b",
        help="Model name or alias (qwen2-2b, llava-7b, llama-11b, or full HF model path)",
    )
    p.add_argument(
        "--spatial_holdout_frac",
        type=float,
        default=0.15,
        help="Fraction of object pairs to reserve as a spatial holdout split (0 disables).",
    )
    p.add_argument(
        "--spatial_holdout_seed",
        type=int,
        default=42,
        help="Random seed for spatial holdout pair sampling.",
    )

    # Budgeted selection flags
    p.add_argument(
        "--param_budget_frac",
        type=float,
        default=0.01,
        help="Fraction of TOTAL params to train.",
    )
    p.add_argument(
        "--selector",
        type=str,
        default="lm_mlp_lastk",
        choices=[
            "lm_mlp_lastk",
            "lm_attn_lastk",
            "lm_norms_lastk",
            "lm_head_only",
            "vision_mlp_lastk",
            "vision_attn_lastk",
            "vision_norms_lastk",
        ],
        help="Selector strategy. Use 'lm_*' for language model or 'vision_*' for vision encoder.",
    )
    p.add_argument(
        "--last_k_blocks",
        type=int,
        default=6,
        help="How many last blocks to consider (for LM or vision depending on selector).",
    )
    p.add_argument(
        "--bitfit_bias_only",
        action="store_true",
        help="Train only bias tensors (BitFit).",
    )
    p.add_argument(
        "--no_lm_head",
        action="store_true",
        help="Do NOT include lm_head in the candidate set (for LM selectors).",
    )
    p.add_argument(
        "--no_projector",
        action="store_true",
        help="Do NOT include multimodal projector parameters in the candidate set (for vision selectors).",
    )

    # Optimization & runtime
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)  # cannot be smaller than 2
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--adam8bit", action="store_true")

    # Eval/log
    p.add_argument("--eval_max_samples", type=int, default=None)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument(
        "--eval_save_steps",
        type=int,
        default=2000,
        help="Run evaluation and save a checkpoint every N optimizer steps (0 disables).",
    )

    # Spatial ID loss hyperparameters
    p.add_argument(
        "--universal_id_path",
        type=str,
        default=None,
        help="Path to universal spatial ID .pt file (if not provided, spatial ID loss is disabled)",
    )
    p.add_argument(
        "--spatial_id_layer",
        type=int,
        default=11,
        help="Which layer to extract embeddings from for spatial ID loss (default: 11)",
    )
    p.add_argument(
        "--spatial_id_weight",
        type=float,
        default=0.5,
        help="Weight for spatial ID loss component (default: 0.5)",
    )
    p.add_argument(
        "--spatial_id_image_size",
        type=int,
        default=224,
        help="Image size for spatial ID lookup (default: 224)",
    )

    # Wandb arguments
    p.add_argument(
        "--wandb_project",
        type=str,
        default="qwen2vl-spatial-ft",
        help="Wandb project name",
    )
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity/team name",
    )
    p.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    args = p.parse_args()
    train(args)
