# preprocess the raw .h5 file

import pandas as pd
from numba.scripts.generate_lower_listing import description
from scipy.io import mmread
import scanpy as sc
import numpy as np
import time
import torch
import random
from numba.scripts.generate_lower_listing import description
from transformers import BertTokenizer,BertModel,LongformerModel,LongformerTokenizer
import json
import os
from tqdm import tqdm
import requests
import json

dataset_name="norman"
gene_num=5000

def sliding_window_split(sequence, window_size=4096, stride=4096):
    num_tokens = len(sequence)
    segments = []
    for i in range(0, num_tokens, stride):
        segment = sequence[i:i + window_size]
        segments.append(segment)
        if len(segment) < window_size:
            break  # 如果最后一个片段小于 window_size，则停止
    return segments

def extract_gene_name(text):
    filtered_parts = [part for part in text.split('+') if part != "ctrl"]
    filtered_parts.sort()
    return filtered_parts


def extract_perturbation(condition):
    ctrl=0
    single_perturb=0
    double_perturb=0
    gene_single=[] # single perturbation gene
    gene_double=[] # single perturbation gene
    for i in range(len(condition)):
        t=extract_gene_name(condition[i])
        if len(t)==0:
            ctrl+=1
        elif len(t)==1:
            single_perturb+=1
            gene_single += t
        elif len(t)==2:
            double_perturb+=1
            gene_double+=t



    print()
    print("The proportion of each type of disturbance relative to the total")
    print("ctrl num: "+str(ctrl))
    print("single_perturb: "+str(len(set(gene_single))))
    print("double_perturb: "+str(len(set(gene_double))))
    if double_perturb==0:
        perturb_type="single"
    else:
        perturb_type="double"
    return list(set(gene_single)),perturb_type


def get_gene_info_ensembl(gene_symbols):
    gene_id_dataset = pd.read_csv("./data/genes_"+dataset_name+".tsv", sep='\s+', header=None)

    base_url_symbol = "https://rest.ensembl.org/lookup/symbol/human/{gene_symbol}?content-type=application/json"
    base_url_id = "https://rest.ensembl.org/lookup/id/{gene_id}?content-type=application/json"

    results = {}
    failed_genes_name = []  # Record failed queries by name
    failed_genes=[] # Record failed queries by name and id
    coding_num=0
    # Step 1: First attempt with gene symbols in batch
    for gene_symbol in gene_symbols:
        url = base_url_symbol.format(gene_symbol=gene_symbol)
        response = requests.get(url)

        if response.status_code == 200:
            gene_info = response.json()
            biotype = gene_info.get('biotype', '').lower()
            is_coding = biotype == 'protein_coding'

            results[gene_symbol] = {
                'is_coding': is_coding,
                'biotype': biotype
            }
            if is_coding==1:
                coding_num+=1
        else:
            # Failed, add to failed list
            failed_genes_name.append(gene_symbol)

        # Avoid request overloading with a small delay
        time.sleep(0.5)

    # Step 2: Use Gene IDs for those that failed
    if failed_genes_name:
        # Filter dataset to get gene IDs for failed gene symbols
        failed_gene_ids = gene_id_dataset[gene_id_dataset.iloc[:, 1].isin(failed_genes_name)].iloc[:, 0].tolist()

        for gene_name in failed_genes_name:
            gene_id = gene_id_dataset[gene_id_dataset.iloc[:, 1].isin([gene_name])].iloc[:, 0].tolist()

            if gene_id==[]:
                print(gene_id+" can't be found")

            url = base_url_id.format(gene_id=gene_id[0])
            response = requests.get(url)

            if response.status_code == 200:

                gene_info = response.json()
                biotype = gene_info.get('biotype', '').lower()
                is_coding = biotype == 'protein_coding'

                # Update results for these failed genes
                results[gene_name] = {
                    'is_coding': is_coding,
                    'biotype': biotype
                }

                if is_coding==1:
                    coding_num+=1
            else:
                # If ID lookup also fails, mark as failed
                failed_genes.append(gene_id)
                results[gene_id] = {'is_coding': None, 'biotype': None}

            # Avoid request overloading with a small delay
            time.sleep(0.5)
    print("coding num: "+str(coding_num)+"/"+str(len(gene_symbols)))
    return results, failed_genes


def batch_get_gene_info_ensembl(gene_symbols, gene_id_file="./data/genes_adamson.tsv", batch_size=50):


    # 读取基因ID数据集
    gene_id_dataset = pd.read_csv(gene_id_file, sep='\s+', header=None)
    gene_id_map = dict(zip(gene_id_dataset.iloc[:, 1], gene_id_dataset.iloc[:, 0]))  # 基因名 -> 基因ID映射
    id_gene_map=dict(zip(gene_id_dataset.iloc[:, 0], gene_id_dataset.iloc[:, 1]))

    gene_ids = [gene_id_map.get(symbol, None) for symbol in gene_symbols]
    # [i for i, x in enumerate(gene_ids) if x is None]

    server = "https://rest.ensembl.org"
    ext_lookup = "/lookup/id"
    ext_sequence = "/sequence/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    results_lookup = {}
    results_sequence={}
    failed_genes = []  # miss RNA
    coding_count = 0
    for i in range(0, len(gene_ids), batch_size):
        print(i)
        batch_ids = gene_ids[i:i + batch_size]
        data = {"ids":batch_ids
        }
        data_json = json.dumps(data)
        response_lookup = requests.post(server+ext_lookup, headers=headers, data=data_json)
        response_sequence = requests.post(server+ext_sequence, headers=headers, data=data_json)

        if response_lookup.status_code == 200 and response_sequence.status_code==200:
            response_lookup = response_lookup.json()
            response_sequence=response_sequence.json()

            failed_genes.extend( (set(batch_ids)-set(list(response_lookup.keys()))) ) #remove shot gene from list

            for gene_id,gene_info in response_lookup.items():
                if gene_info is not None:
                    description=gene_info['description']
                    gene_name=id_gene_map[gene_id]
                    biotype=gene_info['biotype']
                    is_coding = biotype == 'protein_coding'

                    results_lookup[gene_name] = {
                        'is_coding': is_coding,
                        'biotype': biotype,
                        'description':description
                    }
                    coding_count += is_coding
                else:
                    print(f"Gene ID lookup failed: {id_gene_map[gene_id]}")
                    failed_genes+=[id_gene_map[gene_id]]

            for seq_info in response_sequence:
                if seq_info is not None:
                    sequence=seq_info['seq']
                    gene_id=seq_info['id']
                    gene_name=id_gene_map[gene_id]
                    results_sequence[gene_name] = {
                        'sequence':sequence
                    }
                else:
                    print(f"Gene ID sequence failed: {gene_id}")
                    failed_genes+=[gene_id]
        else:
            print(f"Batch ID lookup failed: {batch_ids}")
            failed_genes+=batch_ids

        # 避免请求频率过高
        time.sleep(1)

    print(f"Protein-coding genes: {coding_count}/{len(gene_symbols)}")
    print(f"non-coding genes: {len(gene_symbols)-coding_count-len(failed_genes)}/{len(gene_symbols)}")
    print(f"Not found genes: {len(failed_genes)}/{len(gene_symbols)}")
    return results_lookup,results_sequence, failed_genes

def get_gene_info_NCBI(gene_names, batch_size=10):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    all_results = []

    for i in range(0, len(gene_names), batch_size):
        # 当前批次的基因
        batch_genes = gene_names[i:i + batch_size]

        # 将批次基因连接成一个查询字符串
        term = ",".join(batch_genes)

        # 设置请求参数
        params = {
            'db': 'gene',
            'term': term,
            'retmode': 'json'
        }

        # 发送请求并获取响应
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            result = response.json()
            gene_ids = result.get('esearchresult', {}).get('idlist', [])
            all_results.extend(gene_ids)
        else:
            print(f"Error fetching data for batch {batch_genes}")

        # 延迟一段时间，避免请求过于频繁
        time.sleep(1)  # 可以调整这个时间间隔，根据实际情况

    return all_results


def select_target_gene(adata, genes_to_keep, target_num=3000):
    sc.pp.highly_variable_genes(adata, n_top_genes=target_num)
    highly_variable_genes = adata.var[adata.var['highly_variable']]['gene_name'].tolist()

    final_genes = list(genes_to_keep)
    remaining_genes = [gene for gene in highly_variable_genes if gene not in genes_to_keep]

    num_to_select = target_num - len(final_genes)
    if num_to_select > 0:
        final_genes.extend(remaining_genes[:num_to_select])

    final_genes = final_genes[:target_num]
    filtered_genes = adata.var[adata.var['gene_name'].isin(final_genes)].index
    adata = adata[:, filtered_genes]
    return adata



if __name__ == "__main__":
    folder_path = "result/"+dataset_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"floder '{folder_path}' has been created.")
    else:
        print(f"floder '{folder_path}' already exits.")


    adata=sc.read_h5ad("data/"+dataset_name+".h5ad")

    file_path="data/genes_"+dataset_name+".tsv"
    if not os.path.exists(file_path):
        adata.var.to_csv(file_path, index=True, header=False,sep=" ")
        print(f"file '{file_path}' has been created.")
    else:
        print(f"file '{file_path}' already exits.")

    perturb_gene,perturb_type=extract_perturbation(adata.obs['condition'].unique())
    # perturb_type is "single" or "double"
    print()
    print("num of perturb genes")
    print(len(perturb_gene))

    # adata.obs.drop_duplicates(subset='perturbation')[['nperts', 'perturbation']]

    # res_perturb,miss_perturb=get_gene_info_ensembl(perturb_gene)

    res_lookup,res_seq,miss_perturb=batch_get_gene_info_ensembl(perturb_gene,gene_id_file="./data/genes_"+dataset_name+".tsv")

    # res_perturb=get_gene_info(perturb_gene)

    # save perturbation gene
    with open('./result/'+dataset_name+'_'+str(gene_num)+'/perturb_gene_info.json', 'w') as json_file:
        json.dump(res_lookup, json_file, indent=4)

    with open('./result/'+dataset_name+'_'+str(gene_num)+'/perturb_gene_seq.json', 'w') as json_file:
        json.dump(res_seq, json_file, indent=4)

    with open('./result/'+dataset_name+'_'+str(gene_num)+'/perturb_miss.json', 'w') as json_file:
        json.dump(miss_perturb, json_file, indent=4)

    filtered_data=select_target_gene(adata,perturb_gene,target_num=gene_num)
    print(filtered_data.shape)

    res_all_lookup,res_all_seq,miss_gene=batch_get_gene_info_ensembl(filtered_data.var['gene_name'].values,gene_id_file="./data/genes_"+dataset_name+".tsv")
    with open('./result/'+dataset_name+'_'+str(gene_num)+'/target_gene_info.json', 'w') as json_file:
        json.dump(res_all_lookup, json_file, indent=4)

    with open('./result/'+dataset_name+'_'+str(gene_num)+'/target_gene_seq.json', 'w') as json_file:
        json.dump(res_all_seq, json_file, indent=4)

    with open('./result/'+dataset_name+'_'+str(gene_num)+'/target_miss.json', 'w') as json_file:
        json.dump(miss_gene, json_file, indent=4)