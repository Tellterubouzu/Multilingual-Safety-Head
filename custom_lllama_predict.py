import torch
from lib.utils.custommodel import CustomLlamaModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig

class Custom_llama():
    def __init__(self,model_path, quant_type=None, use_sys_prompt=False, sys_prompt="", generation_config=None,use_history = False):
        model_name = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token =self.tokenizer.eos_token
        self.use_history = use_history
        if quant_type == None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map = "auto",
                torch_dtype = torch.bfloat16,
            )
        elif quant_type == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype = torch.bfloat16,
                quantization_config = quantization_config,
                low_cpu_mem_usage = True,
                trust_remote_code = True,
            )
        elif quant_type == 4:
            bnb_config =BitsAndBytesConfig(load_in_4bit =True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=False,
                                )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype = torch.bfloat16,
                quantization_config = bnb_config,
                low_cpu_mem_usage = True,
                trust_remote_code = True,
            )
            
        if use_sys_prompt:
            self.make_prompt("system", sys_prompt)
        self.default_generation_config = {
            "do_sample": True,
            # "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.8,
            "max_new_tokens": 2048
        }
        self.generation_config = generation_config if generation_config else self.default_generation_config

    def make_prompt(self, role, content):
        self.chat.append({"role": role, "content": content})
    def merge_configs(self, custom_config):
        config = self.default_generation_config.copy()
        if custom_config:
            config.update(custom_config)
        return config
    def predict(self, input, generation_config=None):
        config = self.merge_configs(generation_config)
        self.make_prompt("user", input)
        prompt = self.tokenizer.apply_chat_template(self.chat, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        if config["do_sample"]==True:
            with torch.no_grad():
                outputs = self.model.generate(
                    **input_ids,
                    do_sample=config["do_sample"],
                    #top_k=config["top_k"],
                    top_p=config["top_p"],
                    temperature=config["temperature"],
                    max_new_tokens=config["max_new_tokens"]
                )
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **input_ids,
                    do_sample = False
                )
        generated = self.tokenizer.decode(outputs[0][len(input_ids['input_ids'][0]):], skip_special_tokens=True)
        if self.use_history == True:
            self.make_prompt("assistant",generated)
        else:        
            self.chat = []
        return generated

# 使用例
if __name__ == "__main__":
    llama = Custom_llama(model_path= "models/Llama3.2-3B-Instruct-run_test",quant_type=None, use_sys_prompt=True, sys_prompt="You are a pirate chatbot who always responds in pirate speak!")
    response = llama.predict("Who are you?")
    print(response)