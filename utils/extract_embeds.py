from sys import path
from transformers import (
    AutoProcessor,
    InternVLForConditionalGeneration,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    MllamaForConditionalGeneration,
    Gemma3ForConditionalGeneration,
)
import torch
import argparse


class VLMForExtraction:
    @staticmethod
    def get_model_string(model: str) -> str:
        mapping = {
            "llava-7b": "llava-hf/llava-1.5-7b-hf",
            "llava-13b": "llava-hf/llava-1.5-13b-hf",
            "qwen2-2b": "Qwen/Qwen2-VL-2B-Instruct",
            "qwen-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
            "llama-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "llava-cot": "Xkev/Llama-3.2V-11B-cot",
            "internvl-1b": "OpenGVLab/InternVL3-1B-hf",
            "internvl-2b": "OpenGVLab/InternVL3-2B-hf",
            "internvl-8b": "OpenGVLab/InternVL3-8B-hf",
            "internvl-14b": "OpenGVLab/InternVL3-14B-hf",
            "gemma-4b": "google/gemma-3-4b-it",
            "gemma-12b": "google/gemma-3-12b-it",
        }
        try:
            return mapping[model]
        except KeyError:
            raise ValueError(f"Unknown model alias: {model}")

    def __init__(self, model, device="cuda"):
        self.model_name = model
        self.dtype = None
        model_str = self.get_model_string(model)

        if model == "llava-7b":
            self.layer_indices = list(range(33))
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=device,
            )
            self.processor = AutoProcessor.from_pretrained(model_str)
            self.language_model = self.model.model.language_model
        elif model == "llava-13b":
            self.layer_indices = list(range(41))
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=device,
            )
            self.processor = AutoProcessor.from_pretrained(model_str)
            self.language_model = self.model.model.language_model

        elif model == "qwen2-2b":
            self.layer_indices = list(range(29))
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_str, torch_dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_str)
            self.language_model = self.model.model.language_model

        elif model == "qwen-7b":
            self.layer_indices = list(range(29))
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.float16,
                device_map=device,
            )
            self.processor = AutoProcessor.from_pretrained(model_str)
            self.language_model = self.model.model.language_model

        elif model == "qwen-3b":
            self.layer_indices = list(range(37))
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.float16,
                device_map=device,
                attn_implementation="sdpa",
            )
            self.processor = AutoProcessor.from_pretrained(model_str)
            self.language_model = self.model.model.language_model

        elif model == "llama-11b":
            self.layer_indices = list(range(33))
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            self.processor = AutoProcessor.from_pretrained(model_str)
            self.language_model = self.model.model.language_model

        elif model == "llava-cot":
            self.layer_indices = list(range(41))
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            self.language_model = self.model.model.language_model
            self.processor = AutoProcessor.from_pretrained(model_str)

        elif model == "internvl-1b":
            self.layer_indices = list(range(25))
            self.model = InternVLForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device,
                attn_implementation="sdpa",
            )
            self.processor = AutoProcessor.from_pretrained(
                model_str, trust_remote_code=True
            )
            self.dtype = torch.bfloat16
            self.language_model = self.model.language_model

        elif model == "internvl-2b":
            self.layer_indices = list(range(29))
            self.model = InternVLForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device,
                attn_implementation="sdpa",
            )
            self.processor = AutoProcessor.from_pretrained(
                model_str, trust_remote_code=True
            )
            self.dtype = torch.bfloat16
            self.language_model = self.model.language_model

        elif model == "internvl-8b":
            self.layer_indices = list(range(29))
            self.model = InternVLForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device,
                attn_implementation="sdpa",
            )
            self.processor = AutoProcessor.from_pretrained(
                model_str, trust_remote_code=True
            )
            self.dtype = torch.bfloat16
            self.language_model = self.model.language_model

        elif model == "internvl-14b":
            self.layer_indices = list(range(49))
            self.model = InternVLForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=device,
                attn_implementation="sdpa",
            )
            self.processor = AutoProcessor.from_pretrained(
                model_str, trust_remote_code=True
            )
            self.dtype = torch.bfloat16
            self.language_model = self.model.language_model
        elif model == "gemma-4b":
            self.layer_indices = list(range(30))
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_str, trust_remote_code=True
            )
            self.dtype = torch.bfloat16
            self.language_model = self.model.language_model
        elif model == "gemma-12b":
            self.layer_indices = list(range(40))
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_str,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_str, trust_remote_code=True
            )
            self.dtype = torch.bfloat16
            self.language_model = self.model.language_model

        else:
            raise ValueError(f"Unsupported model: {model}")

        self.device = device

    def _generate(self, text, image, interventions, num_generate=0):
        """
        interventions: dict[layer_idx] → list of (token_str, nth_occurrence, func) tuples
        - layer_idx: int, which transformer block to hook
        - token_str: str, the exact token text as from convert_ids_to_tokens
        - nth_occurrence: int,  1-based index of which occurrence to intervene on
        - func: Callable[[Tensor], Tensor], maps
                activation_slice: (batch, hidden_size) → new_activation_slice
        """

        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": text}],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[text], return_tensors="pt")

        inputs = inputs.to(self.device)
        if self.dtype is not None:
            inputs = inputs.to(self.dtype)
        input_ids = inputs["input_ids"]  # shape (1, prompt_len)

        input_tokens = self._convert_ids_to_tokens(input_ids[0].tolist())
        # print(len(input_tokens))

        # Register one hook per layer, capturing *the same* `tokens` list
        handles = []
        for layer_idx, hooks in interventions.items():
            block = self.language_model.layers[layer_idx - 1]

            def hook(module, _inputs, output):
                hidden_states, *rest = output

                mod_hidden = hidden_states.clone()
                # NOTE: Currently we only support intervention in the initial input query
                # NOTE: Must have cache turned on for this to work
                if mod_hidden.shape[1] == len(input_tokens):
                    for token_str, nth, func in hooks:
                        positions = [
                            i for i, t in enumerate(input_tokens) if t == token_str
                        ]
                        if len(positions) > nth:
                            idx = positions[nth]
                            slice_ = mod_hidden[:, idx, :]
                            mod_hidden[:, idx, :] = func(slice_)

                return (mod_hidden, *rest)

            handles.append(block.register_forward_hook(hook, prepend=True))

        with torch.no_grad():
            if num_generate > 0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=num_generate,
                    output_hidden_states=True,
                    output_logits=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
            else:
                outputs = self.model(**inputs, output_hidden_states=True)

        for h in handles:
            h.remove()

        return inputs, outputs

    def _convert_ids_to_tokens(self, input_ids):
        return self.processor.tokenizer.convert_ids_to_tokens(input_ids)

    def extract_embeds(
        self,
        text,
        image,
        target_tokens,
        last_only: bool = True,
        num_generate: int = 0,
        interventions={},
        return_outputs: bool = False,
    ):
        inputs, outputs = self._generate(
            text, image, interventions=interventions, num_generate=num_generate
        )

        # print(len(outputs.hidden_states))

        # pick up the right input_ids (include generated tokens if any)
        if hasattr(outputs, "sequences"):
            input_ids = outputs.sequences[0]
        else:
            input_ids = inputs["input_ids"][0]
        input_tokens = self._convert_ids_to_tokens(input_ids)

        layer_embeds = {layer: {} for layer in self.layer_indices}
        for token in target_tokens:
            token_indices = [i for i, tok in enumerate(input_tokens) if tok == token]
            if not token_indices:
                print(f"Warning: Token '{token}' not found")
                for layer in self.layer_indices:
                    layer_embeds[layer][token] = None
                continue

            for layer in self.layer_indices:
                if num_generate > 0:
                    layer_steps = []
                    for i, step in enumerate(outputs.hidden_states):
                        # print(step[layer].shape)
                        if i == 0:
                            layer_steps.append(step[layer])
                        else:
                            # print(step[layer][:, -1:, :])
                            layer_steps.append(step[layer][:, -1:, :])
                    hidden_states = torch.cat(layer_steps, dim=1)[0]
                    # print(hidden_states.shape)
                    # print(token_indices)
                    word_embed = hidden_states[token_indices].to(
                        "cpu"
                    )  # shape = (n_occurrences, dim)
                else:
                    try:
                        hidden_states = outputs.hidden_states[layer][0].detach()
                    except Exception as e:
                        print(layer)
                        raise e
                    word_embed = hidden_states[
                        token_indices
                    ].cpu()  # shape = (n_occurrences, dim)
                # print(word_embed.shape)
                if last_only:
                    # keep the single activation (as before)
                    layer_embeds[layer][token] = word_embed[0]
                else:
                    # return all occurrences
                    layer_embeds[layer][token] = word_embed

        if return_outputs:
            return layer_embeds, outputs
        else:
            return layer_embeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embedding")
    parser.add_argument(
        "query",
        type=str,
        help="Text query for embeddings extraction",
    )
    parser.add_argument(
        "target_words",
        type=str,
        help="Comma-separated list of target words (e.g. frog,cat,dog)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the output embeddings",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for extraction",
    )

    args = parser.parse_args()

    target_words = [tw.strip() for tw in args.target_words.split(",") if tw.strip()]
    model = VLMForExtraction(model=args.model)
    target_words = [
        model.processor.tokenizer.tokenize(" " + tw)[-1] for tw in target_words
    ]
    embed = model.extract_embeds(args.query, None, target_words)
    torch.save(embed, args.output_path)
