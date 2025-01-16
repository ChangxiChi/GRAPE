import torch
import numpy as np
import pandas as pd
import scanpy as sc
import json
import os
import random
import pickle
from torch.utils.data import DataLoader, Dataset


class PertData:
    def __init__(self,data_name="adamson",hvg_num=5000,split_ratio=0.75):
        self.hvg_num=hvg_num
        self.testing_cond_path="./result/"+data_name+"_"+str(hvg_num)+"/testing_cond.pkl"
        self.h5_data_path="./data/"+data_name+".h5ad"

        # record the biotype of gene
        with open("./result/"+data_name+"_"+str(hvg_num)+"/target_gene_info.json", 'r') as f:
            self.biotype = json.load(f)

        # gene can't be found
        with open("./result/"+data_name+"_"+str(hvg_num)+"/perturb_miss.json", 'r') as f:
            self.miss_perturbation_gene = json.load(f)

        self.gene_to_id_map=self.get_gene_to_id_map(data_name)
        self.id_to_gene_map=self.get_id_to_gene_map(data_name)


        # coding genes come first, followed by the non-coding genes
        (self.adata, # filtered adata
         self.c_num, # coding gene number
         self.n_num # non-coding gene number
         ) = self.get_adata(data_name)

        self.gene_name=self.adata.var['gene_name'].tolist()

        self.gene_num=self.num_gene()

        self.ctrl=self.get_averger_ctrl()

        # get clustered embedding
        self.desc_embed,self.seq_embed=self.load_embed(data_name)

        self.desc_embed_dim=self.desc_embed.shape[1]
        self.seq_embed_dim = self.seq_embed.shape[1]
        # split dataset
        self.training_cond,self.training_cell,self.testing_cond=self.split_dataset(split_ratio)

        self.DE_gene=self.get_DE_gene(data_name) # differential expression gene

        self.training_set=AnnDataBatchDataset(self.training_cell)


        # train_dataset = AnnDataBatchDataset(train_adata)
        # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        # for batch_data, batch_condition in train_loader:

    def num_gene(self):
        return self.adata.shape[1]

    # return clustered gene
    def cluster_gene(self):
        coding_gene=[]
        noncoding_gene=[]
        for key,values in self.biotype.items():
            if values['is_coding']:
                coding_gene.append(key)
            else:
                noncoding_gene.append(key)

        return coding_gene,noncoding_gene

    # gene name -> Ensembl ID
    def get_gene_to_id_map(self,data_name="adamson"):
        map_path =  "./data/genes_"+data_name+".tsv"
        gene_id_dataset = pd.read_csv(map_path, sep='\s+', header=None)
        gene_id_map = dict(zip(gene_id_dataset.iloc[:, 1], gene_id_dataset.iloc[:, 0]))

        return gene_id_map

    # Ensembl ID -> gene name
    def get_id_to_gene_map(self,data_name="adamson"):
        map_path =  "./data/genes_"+data_name+".tsv"
        gene_id_dataset = pd.read_csv(map_path, sep='\s+', header=None)
        id_gene_map = dict(zip(gene_id_dataset.iloc[:, 0], gene_id_dataset.iloc[:, 1]))

        return id_gene_map

    # Target genes including perturbation genes and HVGs
    # Please run Proprocess.py at first
    # Target genes are selected and stored in the program's results.
    # Remove cells where perturbation information cannot be found
    def get_adata(self,data_name="adamson"):
        adata_path="./data/"+data_name+".h5ad"
        adata=sc.read_h5ad(adata_path)

        # select target genes
        gene_list=list(self.biotype.keys())

        # use gene name to select adata
        adata = adata[:, adata.var['gene_name'].isin(gene_list)]

        coding_gene,noncoding_gene=self.cluster_gene()

        gene_index=[np.where(adata.var['gene_name']==gene)[0][0] for gene in coding_gene + noncoding_gene]

        # adata.var = adata.var.set_index('gene_name').reindex(coding_gene+noncoding_gene).reset_index(drop=False)
        adata = adata[:,gene_index]

        # add idx
        adata.var.loc[:,'idx'] = range(adata.shape[1])

        # remove cells where perturbation information cannot be found
        cond_to_remove = [item + '+ctrl' for item in self.miss_perturbation_gene]
        adata = adata[~adata.obs['condition'].isin(cond_to_remove), :]

        return adata,len(coding_gene),len(noncoding_gene)

    # get the embedding of Gene Sequence and Description
    # result of Process.py
    def load_embed(self,data_name="adamson"):
        seq_path = "./result/"+data_name+"_"+str(self.hvg_num)+"/seq_embed.json"
        desc_path = "./result/"+data_name+"_"+str(self.hvg_num)+"/desc_embed.json"

        with open(desc_path, 'r') as f:
            desc_dict = json.load(f)

        with open(seq_path, 'r') as f:
            seq_dict = json.load(f)

        desc_embed=[]
        seq_embed=[]
        for gene_name in self.gene_name:
            desc=desc_dict[gene_name][0]
            seq=seq_dict[gene_name]
            desc_embed.append(desc)
            seq_embed.append(seq)
        return torch.tensor(desc_embed),torch.tensor(seq_embed)

    # split dataset based on perturbations
    # return training set and testing set
    def split_dataset(self,ratio=0.75,only_single=True):
        if os.path.exists(self.testing_cond_path):
            with open(self.testing_cond_path, 'rb') as f:
                data = pickle.load(f)
            print(data)

            single_perturb=[]
            perturbation_list = list(self.adata.obs['condition'].unique())
            for cond in perturbation_list:
                t = self.extract_gene_name(text=cond)
                if len(t) <= 1:
                    single_perturb += [cond]

            perturbation_list = list(set(single_perturb))

            perturbation_list.remove('ctrl')
            test_cond=data
            train_cond=[item for item in perturbation_list if self.extract_gene_name(item)[0] not in test_cond]
            train_cond.append('ctrl')

            # split adata
            train_cell=self.adata[self.adata.obs['condition'].isin(train_cond),:]
            # test_cell=self.adata[self.adata.obs['condition'].isin(test_cond),:]

            return train_cond, train_cell, test_cond

        else:
            if only_single:
                single_gene=[] # record single perturbation gene
                perturbation_list=list(self.adata.obs['condition'].unique())
                for cond in perturbation_list:
                    t = self.extract_gene_name(text=cond)
                    if len(t)==1:
                        single_gene+=t

                single_gene=list(set(single_gene))

                # remove 'ctrl' condition
                # 'ctrl' represent unperturbed state of cell
                if 'ctrl' in single_gene:
                    single_gene.remove('ctrl')
                random.shuffle(single_gene)
                train_size=int(len(single_gene)*ratio)
                train_cond=single_gene[:train_size]
                test_cond=single_gene[train_size:]

                # split adata
                train_idx=[]
                test_idx=[]
                for i in range(len(list(self.adata.obs['condition']))):
                    cond = self.adata.obs['condition'][i]
                    t = self.extract_gene_name(text=cond)
                    if (len(t)==1 and (t[0] in train_cond)) or len(t)==0: #including ctrl
                        train_idx.append(i)
                    elif len(t)==1 and (t[0] in test_cond):
                        test_idx.append(i)



                train_cell=self.adata[train_idx,:]
                test_cell=self.adata[test_idx,:]

                # save testing cond for prediction
                with open(self.testing_cond_path, 'wb') as f:
                    pickle.dump(test_cond, f)

                return train_cond,train_cell,test_cond

    def get_id_gene(self,id_list):
        gene_list=[]
        for id in id_list:
            gene=self.id_to_gene_map[id]
            gene_list.append(gene)

        return gene_list

    # input index
    # output gene name
    def get_idx_gene(self,idx_list):
        gene_list=[]
        for idx in idx_list:
            gene=self.adata.var[self.adata.var['idx'] == idx]['gene_name'].values[0]
            gene_list.append(gene)

        return gene_list

    # return index of gene
    def get_gene_idx(self,gene_list):
    # gene can be gene name
        idx_list=[]
        for gene in gene_list:
            if gene in list(self.adata.var['gene_name']):
                # input is gene name
                idx=self.adata.var[self.adata.var['gene_name']==gene]['idx'].values[0]
                idx_list.append(idx)

            else:
                print(gene+" can't be found!")

        return idx_list

    # obtain biotype of gene
    # coding or non-coding
    def get_biotype(self,name_list):
        biotype_list=[]
        for name in name_list:
            if self.biotype[name]['is_coding']:
                biotype='coding'
                biotype_list.append(biotype)
            else:
                biotype='non-coding'
                biotype_list.append(biotype)

        return biotype_list

    def get_averger_ctrl(self):
        ctrl = self.adata[self.adata.obs['condition'].isin(['ctrl']), :]
        control = torch.tensor(np.mean(ctrl.X.toarray(), axis=0))
        return control

    def extract_gene_name(self,text):
        filtered_parts = [part for part in text.split('+') if part != "ctrl"]
        filtered_parts.sort()
        return filtered_parts

    def get_DE_gene(self,data_name,top_de=40):
        DE_gene=[]
        for perturb in self.testing_cond:
            # perturbation key of DE dict
            if data_name == "adamson":
                de_key = 'K562(?)_' + perturb + '+ctrl_1+1'
            elif data_name == 'norman':
                if 'A549_' + perturb + '+ctrl_1+1' in self.adata.uns['rank_genes_groups_cov_all'].keys():
                    de_key = 'A549_' + perturb + '+ctrl_1+1'
                else:
                    de_key = 'A549_ctrl+' + perturb + '_1+1'

            id_list = self.adata.uns['rank_genes_groups_cov_all'][de_key]
            gene_name = self.get_id_gene(list(id_list))

            gene_name = [gene for gene in gene_name if gene in self.gene_name][:top_de]
            DE_gene = DE_gene + gene_name

        DE_gene=list(set(DE_gene))
        return self.get_gene_idx(DE_gene)


class AnnDataBatchDataset(torch.utils.data.Dataset):
    def __init__(self, adata, batch_size=32):
        self.adata = adata
        self.batch_size = batch_size
        self.num_cells = adata.shape[0]

        self.X = adata.X
        self.condition = adata.obs['condition']

    def __len__(self):
        return self.num_cells

    def __getitem__(self, idx):
        cell_data = self.X[idx, :]
        cell_condition = self.condition.iloc[idx]  # obtain cell's condition

        return torch.tensor(cell_data.toarray(), dtype=torch.float32), cell_condition

class MultiModalDataset(Dataset):
    def __init__(self, desc_embed, seq_embed):
        self.desc_embed = desc_embed
        self.seq_embed = seq_embed

    def __len__(self):
        return self.desc_embed.shape[0]

    def __getitem__(self, idx):
        return self.desc_embed[idx], self.seq_embed[idx]


class MyDataset(Dataset):
    def __init__(self, desc_data, seq_data):
        self.desc_data = desc_data
        self.seq_data = seq_data

    def __len__(self):
        return len(self.desc_data)

    def __getitem__(self, idx):
        desc = self.desc_data[idx]
        seq = self.seq_data[idx]
        return desc, seq

if __name__ == "__main__":
    pertdata=PertData()