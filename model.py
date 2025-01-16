import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.ao.nn.quantized.functional import threshold
from tqdm import tqdm

from modules import *
import pytorch_lightning as pl
from pertdata import MyDataset
from utils import *
import torch.nn as nn
import dgl
from torch.utils.data import Dataset, DataLoader


class model(pl.LightningModule):
    def __init__(self,
                 perturb_data,
                 seq_embed_dim=256,
                 desc_embed_dim=768,
                 hidden_dim=256,
                 hidden_feats=256,
                 out_feats=64,
                 num_heads=4,
                 k_hop=3,
                 k_hop_feats=[256,256,256,64],
                 threshold_desc=0.6,
                 threshold_seq=0.6,
                 threshold_fusion=0.6,
                 threshold_mmG=0.015,
                 threshold_k_hop=0.04,
                 threshold_final=0.015,
                 Lambda=1):
        '''

        :param seq_embed_dim:
        :param desc_embed_dim:
        :param hidden_dim: output dim of Encoder
        :param hidden_feats: hidden layer dim of HGNN
        :param out_feats: output dim of HGNN
        :param num_rels: num of edge type
        '''
        super(model, self).__init__()

        self.desc_embed=perturb_data.desc_embed.to('cuda')
        self.seq_embed=perturb_data.seq_embed.to('cuda')
        self.c_num=perturb_data.c_num
        self.ctrl=perturb_data.ctrl.to('cuda')
        self.gene_name=perturb_data.adata.var['gene_name']
        self.encoder=Multimodal_Encoder(hidden_dim=hidden_dim,desc_embed_dim=desc_embed_dim,seq_embed_dim=seq_embed_dim)


        self.GRN_network=HeterGAT(layer_dim=[hidden_dim,hidden_dim,out_feats],num_heads=num_heads,edge_type=['CC','CN','NN'], dropout_ft=0.4,dropout_att=0.4)
        self.fusion=GraphChannelAttLayer(num_channel=3)
        self.fusion_final=GraphChannelAttLayer(num_channel=k_hop)

        # self.multiGNN=MultiLayerGCN(in_feats=hidden_dim,hidden_feats=hidden_feats,out_feats=out_feats,num_layers=k_hop)
        self.multiGNN=MultiLayerGCN(feats=k_hop_feats,num_layers=k_hop)
        self.MultiLayer_simHead=nn.ModuleList([MultiHeadSimilarity(k_hop_feats[i+1]) for i in range(k_hop)])
        # gene specific decoder
        gene_num=perturb_data.gene_num
        self.indv_w1 = nn.Parameter(torch.rand(gene_num,out_feats, 1))
        self.indv_b1 = nn.Parameter(torch.rand(gene_num, 1))

        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)


        self.norm = nn.BatchNorm1d(out_feats)
        self.gene_decoder=MLP([out_feats,2*out_feats,out_feats])
        self.relu=nn.ReLU()

        self.threshold_desc=threshold_desc
        self.threshold_seq=threshold_seq
        self.threshold_fusion=threshold_fusion
        self.threshold_final=threshold_final
        self.threshold_k_hop=threshold_k_hop
        self.threshold_mmG = threshold_mmG

        self.Lambda =Lambda


    def train_encoder(self,epoch=10):
        dataset=MyDataset(self.desc_embed,self.seq_embed)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        for epoch in range(epoch):
            with tqdm(dataloader, desc=f"Pre-training Encoder, Epoch {epoch + 1}", unit="batch") as pbar:
                for desc, seq in pbar:
                    loss = self.encoder.alignment(desc.to('cuda'), seq.to('cuda'))
                    pbar.set_postfix(loss=loss)

    # construct multi-modal graph
    # view as a homogeneous graph
    def make_mmG(self,f_mm,f_desc,f_seq):
        with torch.no_grad():
            # obtain similarity matrix of three modalities
            sim_desc=cosine_similarity(f_desc)
            sim_seq=cosine_similarity(f_seq)
            sim_mm=cosine_similarity(f_mm)

            # construct graph
            sim_desc[sim_desc < self.threshold_desc] = 0
            sim_seq[sim_seq < self.threshold_seq] = 0
            sim_mm[sim_mm < self.threshold_fusion] = 0

            graph=torch.mean(torch.stack([sim_desc,sim_seq,sim_mm]),dim=0)
            graph=F.normalize(graph)
            graph[graph<self.threshold_mmG]=0
            # graph=torch.where(graph>self.threshold_mmG,graph,zero)


        return graph

    # get three graphs that only contain 'C-C', 'N-N' and 'C-N' edge respectively
    # input multi-modal graph
    def get_sub_graph(self,graph):
        with torch.no_grad():
            c_num=self.c_num

            # coding gene first
            g_cc = graph.clone()
            g_nn = graph.clone()
            g_cn = graph.clone()

            zero=torch.tensor(0.0,device=self.device)

            g_cc[c_num:, :] = zero
            g_cc[:, c_num:] = zero

            g_nn[:c_num, :] = zero
            g_nn[:, :c_num] = zero

            g_cn[:c_num, :c_num] = zero
            g_cn[c_num:, c_num:] = zero

        return g_cc,g_cn,g_nn


    # create gene regulation network
    # based on k-hop similarity

    def k_hop_graph(self, graph,X,threshold):
        k_output=self.multiGNN(graph,X,threshold)
        G_list=[]
        for i in range(len(k_output)):
            sim=self.MultiLayer_simHead[i](k_output[i])
            G_list.append(sim)
        zero=torch.tensor(0.0,device=self.device)
        G=self.fusion(G_list)
        G = F.normalize(G)
        G[G < self.threshold_final] = zero
        # G=torch.where(G>self.threshold_final,G,zero)

        return G


    '''
    batch_data: batch_size X gene_num
    '''
    def get_gene_idx(self,gene_list):
    # gene can be gene name
        idx_list=[]
        for gene in gene_list:
            if gene in list(self.gene_name):
                # input is gene name
                idx=np.where(self.gene_name==gene)[0][0]
                idx_list.append(idx)

            else:
                print(gene+" can't be found!")

        return idx_list

    def extract_cond(self, batch_condition):
        cond = []
        idx = []
        for condition in batch_condition:
            if condition == 'ctrl':
                cond.append(condition)
                idx.append(-1)
            else:
                condition = [part for part in condition.split('+') if part != "ctrl"][0]
                idx.append(self.get_gene_idx([condition])[0])
                cond.append(condition)

        return cond, idx

    def training_step(self,batch_data):


        self.desc_embed=self.desc_embed.to(self.device)
        self.seq_embed=self.seq_embed.to(self.device)
        self.ctrl=self.ctrl.to(self.device)

        cond, perturb_idx = self.extract_cond(batch_data[1])

        loss_rec=torch.tensor(0).to(self.device)
        loss_dir=torch.tensor(0).to(self.device)

        f_mm,f_desc,f_seq=self.encoder(desc_embed=self.desc_embed,seq_embed=self.seq_embed)

        # if ignore_ctrl:
        # cond=[item for item in cond if item != "ctrl"]

        cond_type=list(set(cond))

        graph = self.make_mmG(f_mm=f_mm.detach() , f_desc=f_desc.detach() , f_seq=f_seq.detach() )
        G = self.k_hop_graph(graph.detach(), f_mm,threshold=self.threshold_k_hop)
        g_cc,g_cn,g_nn=self.get_sub_graph(G.detach())

        for c in cond_type:
            cond_idx=[i for i, x in enumerate(cond) if x == c]
            gene_idx=perturb_idx[cond_idx[0]] # get idx of perturbation c in current batch
            GT=batch_data[0][cond_idx,:].reshape(len(cond_idx),-1).to(self.device)
            mask = torch.ones_like(f_mm)


            if gene_idx !=-1:
                mask[gene_idx, :] = 0

            masked_gene_embed = f_mm * mask

            # feature of f_cc f_cn f_nn
            f = self.GRN_network([g_cc.detach(),g_cn.detach(),g_nn.detach()],masked_gene_embed)

            f = self.norm(f).squeeze()
            f = self.relu(f)
            f = self.gene_decoder(f)
            f = f.unsqueeze(-1)

            pred = torch.sum(f*self.indv_w1,dim=1)+self.indv_b1
            pred = pred + self.ctrl.unsqueeze(-1)

            pred=pred.reshape(1,-1)
            pred = pred.repeat(len(cond_idx),1)
            loss_rec = loss_rec+torch.sum(torch.mean((pred-GT)**2,dim=1))
            loss_dir = loss_dir+self.dir_loss(pred, GT,self.ctrl.unsqueeze(0))

        loss_rec=loss_rec/len(perturb_idx)
        loss_reg = torch.mean((G - graph.detach()) ** 2)
        loss_dir=loss_dir/len(perturb_idx)
        print("loss_pred: ",loss_rec.item()," loss_reg: ",loss_reg.item(), " loss_dir",loss_dir.item())
        # print("loss_pred: ", loss_rec.item(), " loss_reg: ", loss_reg.item())

        return loss_rec+loss_reg+loss_dir



    def dir_loss(self,pred,GT,ctrl,cluster=False,delta=0.7):
        # MSE, Huber,  cos_similarity

        GT_dir = torch.tanh(GT.squeeze() - ctrl.squeeze())
        pred_dir = torch.tanh(pred - ctrl.reshape(1,-1))
        if cluster:
            loss=torch.mean((pred_dir-GT_dir)**2)
        else:
            loss=torch.sum(torch.mean((pred_dir-GT_dir)**2,dim=1))
        return loss

        # Huber
        # if pert_data is not None:
        #     idx=pert_data.DE_gene
        #     GT=GT[:,idx]
        #     ctrl=ctrl[:,idx]
        #     pred=pred[:,idx]
        #
        # GT_dir = (GT.squeeze() - ctrl.squeeze())
        # pred_dir = (pred - ctrl.reshape(1,-1))
        # error = GT_dir - pred_dir
        # is_small_error = torch.abs(error) <= delta
        # loss = torch.where(is_small_error, 0.5 * error ** 2, delta * (torch.abs(error) - 0.5 * delta))
        # return torch.sum(torch.mean(loss,dim=1))

        # cos
        # GT_dir = (GT.squeeze() - ctrl.squeeze())
        # pred_dir = (pred - ctrl.reshape(1,-1))
        # loss = -F.cosine_similarity(GT_dir,pred_dir,dim=1)
        # return torch.sum(loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3,weight_decay=1e-4)
        return optimizer


    def prediction(self,perturb_idx,threshold=1e-2):
        pred_list = []
        # f_list.append(masked_gene_embed.unsqueeze(0))
        # f_fusion = torch.cat(masked_gene_embed_list, dim=0)

        f_mm,f_desc,f_seq=self.encoder(self.desc_embed.to('cuda'),self.seq_embed.to('cuda'))
        with torch.no_grad():
            graph = self.make_mmG(f_mm=f_mm , f_desc=f_desc , f_seq=f_seq )
        G = self.k_hop_graph(graph, f_mm,threshold=self.threshold_k_hop)

        g_cc,g_cn,g_nn=self.get_sub_graph(G)

        for idx in perturb_idx:
            mask = torch.ones_like(f_mm).to('cuda')
            if idx !=-1:
                mask[idx, :] = -1
                masked_gene_embed = f_mm * mask
            else:
                masked_gene_embed=f_mm

            f = self.GRN_network([g_cc,g_cn,g_nn],masked_gene_embed)
            f = self.norm(f).squeeze()
            f = self.relu(f)
            f = self.gene_decoder(f)
            f = f.unsqueeze(-1)

            pred = torch.sum(f*self.indv_w1,dim=1)+self.indv_b1
            pred = pred + self.ctrl.unsqueeze(-1).to(pred.device)

            # filter negative value
            pred=pred.reshape(1,-1)
            pred[pred<threshold]=0
            pred_list.append(pred)

        prediction=torch.cat(pred_list,dim=0)

        return prediction

    def get_gene_embed(self):
        pred_list = []
        # f_list.append(masked_gene_embed.unsqueeze(0))
        # f_fusion = torch.cat(masked_gene_embed_list, dim=0)

        f_mm,f_desc,f_seq=self.encoder(self.desc_embed.to('cuda'),self.seq_embed.to('cuda'))
        graph = self.make_mmG(f_mm=f_mm , f_desc=f_desc , f_seq=f_seq )
        G = self.k_hop_graph(graph, f_mm,threshold=self.threshold_k_hop)

        g_cc,g_cn,g_nn=self.get_sub_graph(G)

        f = self.GRN_network([g_cc,g_cn,g_nn],f_mm)
        #
        # f = self.norm(f).squeeze()
        # f = self.relu(f)
        # f = self.gene_decoder(f)
        # f = f.unsqueeze(-1)
        #
        # pred = torch.sum(f*self.indv_w1,dim=1)+self.indv_b1
        # pred = pred + self.ctrl.unsqueeze(-1).to(pred.device)


        return self.desc_embed,self.seq_embed,f