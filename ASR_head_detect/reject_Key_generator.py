import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import json
import os
from tqdm import tqdm

class MetaLlama3:
    def __init__(self, model_path=None, quant_type=None, use_sys_prompt=False, sys_prompt="", generation_config=None):
        if model_path is None:
            model_path = "meta-llama/Llama3-3.1-8B-Instruct"
        
        self.model_name = model_path.split("/")[-1]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" 

        if quant_type is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True if quant_type == 8 else False, load_in_4bit=True if quant_type == 4 else False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        self.default_generation_config = {
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.9,
            "max_new_tokens": 256
        }
        self.generation_config = generation_config or self.default_generation_config


    def predict_batch(self, input_texts, generation_config=None, batch_size=8):
        if not isinstance(input_texts, list):
            raise TypeError("predict_batch: input_texts must be a list of strings.")

        config = {**self.default_generation_config, **(generation_config or {})}
        results = []

        for i in range(0, len(input_texts), batch_size):
            batch_inputs = input_texts[i:i + batch_size]

            batch_prompts =[]
            for input_text in batch_inputs:
                local_chat = []
                local_chat.append({"role":"user","content":input_text})
                prompt = self.tokenizer.apply_chat_template(
                    local_chat,
                    tokenize = False,
                    add_generation_prompt=True
                )
                batch_prompts.append(prompt)

            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    top_p=config["top_p"],
                    temperature=config["temperature"],
                    max_new_tokens=config["max_new_tokens"],
                    eos_token_id=[
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    ],
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            for idx, output_tokens in enumerate(outputs):
                input_length = (inputs['attention_mask'][idx] == 1).sum().item()

                generated_text = self.tokenizer.decode(
                    output_tokens[input_length:],
                    skip_special_tokens=True
                )

                results.append(generated_text)

        return results
    

    def moderate_batch(self, input_output_pairs, batch_size=8):
        if not isinstance(input_output_pairs, list):
            raise TypeError("moderate_batch: input_output_pairs must be a list of (input, output) tuples.")

        if "Guard" not in self.model_name:
            raise PermissionError("If you call 'moderate_batch', you must use a 'Guard' model.")

        moderation_results = []

        for i in range(0, len(input_output_pairs), batch_size):
            batch_pairs = input_output_pairs[i:i + batch_size]
            batch_prompts = []
            for user_input, model_output in batch_pairs:
                prompt = f"User Input: {user_input}\nModel Output: {model_output}\nReason:"
                batch_prompts.append(prompt)
            input_ids = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **input_ids,
                    max_new_tokens=100,
                    eos_token_id=[
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    ],
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            for output, (user_input, model_output) in zip(outputs, batch_pairs):
                decoded_output = self.tokenizer.decode(output, skip_special_tokens=True).lower()
                if "unsafe" in decoded_output:
                    moderation_results.append({"input": user_input, "output": model_output, "moderation": "unsafe"})
                else:
                    moderation_results.append({"input": user_input, "output": model_output, "moderation": "safe"})

        return moderation_results

class GenerateBlockKeywords:
    def __init__(self, model_path, dataset_path, moderation=False):
        self.model = MetaLlama3(model_path)
        self.dataset = self.load_dataset(dataset_path)
        self.results = []
        self.moderation = moderation

        if moderation:
            #moderation_path = "~/work/srv11/Multilingual-Safety-Head/models/Llama-Guard-3-8B"
            moderation_path = "../models/Llama-Guard-3-8B"
            
            self.moderator = MetaLlama3(moderation_path)

    def load_dataset(self, dataset_path):
        try:
            data = pd.read_csv(dataset_path)
            return data["input"].tolist()
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset: {e}")

    def save_results_to_csv(self, output_path):
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    def generate(self, output_path, batch_size=15, moderate_batch_size=10):
        total = len(self.dataset)
        intermediate_results = []

        with tqdm(total=total, desc="Generating Responses") as pbar:
            for i in range(0, total, batch_size):
                batch_inputs = self.dataset[i:i + batch_size]
                batch_responses = self.model.predict_batch(batch_inputs, generation_config=None, batch_size=batch_size)

                for input_text, response in zip(batch_inputs, batch_responses):
                    intermediate_results.append((input_text, response))
                    self.results.append({
                        "input": input_text,
                        "response": response,
                        "moderation": None
                    })

                pbar.update(len(batch_inputs))
        
        del self.model
        torch.cuda.empty_cache()

        if self.moderation:
            with tqdm(total=len(intermediate_results), desc="Moderating Responses") as pbar:
                moderation_results = self.moderator.moderate_batch(intermediate_results, batch_size=moderate_batch_size)
                for result, moderation in zip(self.results, moderation_results):
                    result["moderation"] = moderation["moderation"]
                    pbar.update(1)

        self.save_results_to_csv(output_path)

if __name__ == "__main__":
    model_path = "~/work/srv11/Multilingual-Safety-Head/models/Llama-3.1-8B-Instruct"
    model_path = "../models/Llama-3.1-8B-Instruct"
    dataset_path = "../dataset_for_sahara/Multilingual_en_600.csv"
    output_path = "./generated_results_en_600_sample_False.csv"

    generator = GenerateBlockKeywords(model_path, dataset_path, moderation=True)
    generator.generate(output_path, batch_size=10,moderate_batch_size=10)

    print(f"Results saved to {output_path}")
    del generator
    torch.cuda.empty_cache()
    dataset_path = "../dataset_for_sahara/Multilingual_ja_600.csv"
    output_path = "./generated_results_ja_600_sample_False.csv"

    generator = GenerateBlockKeywords(model_path, dataset_path, moderation=True)
    generator.generate(output_path, batch_size=10,moderate_batch_size=10)

    print(f"Results saved to {output_path}")
