import torch
from numba.scripts.generate_lower_listing import description
from transformers import BertTokenizer,BertModel,LongformerModel,LongformerTokenizer

from transformers import AutoModelForSequenceClassification, AutoTokenizer,AutoModelForMaskedLM,AutoModel

checkpoint = "./from_pretrain_DNA_hyena"
max_length = 160_000

tokenizer_DNA = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
DNA = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

checkpoint_bert="./from_pretrain_bert"
tokenizer_bert=AutoTokenizer.from_pretrained(checkpoint_bert)
bert=AutoModel.from_pretrained(checkpoint_bert,output_hidden_states=True).to('cuda')

import json
from tqdm import tqdm

def sliding_window_split(sequence, window_size=max_length, stride=max_length):
    num_tokens = len(sequence)
    segments = []
    for i in range(0, num_tokens, stride):
        segment = sequence[i:i + window_size]
        segments.append(segment)
        if len(segment) < window_size:
            break
    return segments


dataset_name="norman"
hvg_num=5000


info_path="result/"+dataset_name+"_"+str(hvg_num)+"/target_gene_info.json"
seq_path="result/"+dataset_name+"_"+str(hvg_num)+"/target_gene_seq.json"

seq_embed={}
desc_embed={}

with open(info_path,'r') as f:
    info=json.load(f)

with open(seq_path,'r') as f:
    seq=json.load(f)

bert.eval()

with torch.no_grad():
    for key,value in tqdm(info.items()):
        desc=value['description']
        inputs=tokenizer_bert(desc,return_tensors='pt',padding=True).to('cuda')
        output=bert(**inputs)
        embed=output.last_hidden_state[:,0,:]
        desc_embed[key]=embed.tolist()

    with open('./result/'+dataset_name+'_'+str(hvg_num)+'/desc_embed.json', 'w') as json_file:
        json.dump(desc_embed, json_file, indent=4)


DNA.eval()

with torch.no_grad():
    for key, value in tqdm(seq.items()):
        desc = value['sequence']

        embed = torch.zeros((1, 256)).to('cuda')
        segments = sliding_window_split(desc)
        for segment in segments:
            inputs = tokenizer_DNA(segment, return_tensors='pt', padding=True).to('cuda')
            output = DNA(**inputs)

            embed = embed + output.last_hidden_state[:,0,:]

        embed = embed / torch.tensor(len(segments))
        embed =embed.squeeze()
        seq_embed[key] = embed.tolist()

    with open('./result/' + dataset_name + '_'+str(hvg_num)+'/seq_embed.json', 'w') as json_file:
        json.dump(seq_embed, json_file, indent=4)
