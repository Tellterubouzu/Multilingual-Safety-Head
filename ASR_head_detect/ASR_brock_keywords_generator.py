import json
import dataset
import accelerate
import enumulate
import pandas as pd
from meta_llama3 import meta_llama3
import tqdm

from transformers import AutoModelForCausaLM, AutoTokenizer, BitsAndBytesConfig

class generate_block_keywords():
    def __init__(self,model_path:str,dataset:str,moderation=False):

        bnb_config =BitsAndBytesConfig(
            load_in_8bit=True
        )
        try:
            self.model = meta_llama3(model_path,quant_type=8)
        except:
            raise FileNotFoundError(f"{model_path} cant't load ,because of path is not exist.")
        self.datasets =self.load_dataset(dataset_path=dataset)
        
        self.model_name =model_path.split("/")[-1]
        moderation_path = "meta-llama/Llama-Guard-3-8B"
        try:
            self.moderator = meta_llama3(moderation_path,quant_type=8)
        except:
            raise FileNotFoundError(f"{moderation_path} cant't load ,because of path is not exist.")
        


        pass

    
    def load_dataset(self,dataset_path):
        try:
            data_df = pd.read_csv(dataset_path)
            data_df =data_df["input"]
            return data_df
        except:
            raise FileNotFoundError("datset_path is not exist. so I can't load dataset")

    def model_predict(self,text):
        self.model.predict_batch
        output = text
        return output

    def generate(self,debug =True,batch_size=10):
        
        
        if debug :
            total = 50
            self.datasets = self.datasets[:total]
        print(total)
        print(f"Debug is {debug}")
        

        all_prompts =[]
        generated ={}
        file_name = f"{self.model_name}"
        
        with tqdm(total=total, desc=f"Generating Output of:{self.model_name}") as pbar:
            for i in range(0, total, batch_size):
                batch_prompts = all_prompts[i:i+batch_size]
                batch_responses = self.model.predict_batch(batch_prompts)
                for j, response in enumerate(batch_responses):
                    idx = i + j 
                    if idx >= total:
                        break
                    self.save_result(
                        task_id=idx,
                        question=self.datasets[idx],
                        response=response,
                    )
                pbar.update(len(batch_responses))
        
        with tqdm(total = total, desc = f"Moderation output of {self.model_name}") as pbar:
            for i in range(0, total, batch_size=8):
                batch_responses_moderate = 



        self.write_result(file_name,self.model_name,response,moderation)
        return generated
    def save_result(self,task_id,question,response,moderation):
        result_data = {
            "task_id": task_id,
            "question": question,
            "response": response,
        }
        with open(self.result_file_path, 'r+') as f:
            data = json.load(f)
            data.append(result_data)
            f.seek(0)
            json.dump(data, f, indent=4)


if __name__=="__main__":
    model_list = [
        "meta-llama/Llama-3.1-8B-Instruct"
    ]
    dataset_list = [
        "../dataset_for_sahara/Multilingual_en_300.csv"
        "../dataset_for_sahara/Multilingual_ja_300.csv"
    ]

    for model_name in model_list:
        for data in dataset_list:
            generator=generate_block_keywords(model_name,data):
            generator.generate()