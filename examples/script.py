## python file that generates predictions from a pretrained language model from huggingface transformers

import torch
from transformers import  LlamaForCausalLM, LlamaTokenizer
import csv
import pandas as pd
## main function
def main():
    tokenizer = LlamaTokenizer.from_pretrained("/path/llama_hf_7",use_fast=False,cache_dir="/cluster/scratch/username/cache/")
    model = LlamaForCausalLM.from_pretrained("/path/llama_hf_7",device_map="auto",cache_dir="/cluster/scratch/username/cache/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## read prompts from csv and generate predictions
    df=pd.read_csv("./data.csv")
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    with open('./llama007_response.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        row=['pred']
        writer.writerow(row)
    with torch.no_grad():
        for i in range(0,df.shape[0],1):
            prompts=list(df['Prompt'].values)[i:i+1]
            prompts= "You are a normal citizen with average education and intuition.\n"+prompts[0]
            inputs = tokenizer([prompts], return_tensors='pt').to(device)

            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=False,
                max_new_tokens=120,temperature=0,eos_token_id= tokenizer.eos_token_id
            )
            outputs=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            outputs=[[el] for el in outputs]
            with open('./llama007_response.csv', 'a') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                writer.writerows(outputs)
## call main function
if __name__ == '__main__':
    main()
