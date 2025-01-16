import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn
from dgl.nn.pytorch import RelGraphConv,GraphConv
import dgl
import dgl.nn.pytorch as dglnn
import torch.optim as optim
from dgl.nn.pytorch import GATConv

from utils import cosine_similarity


# # heterogeneous graph neural network
# class HGNN_Model(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  hidden_feats,
#                  out_feats,
#                  num_rels=3,
#                  device='cuda'):
#         '''
#         :param in_feats: input dimension
#         :param hidden_feats: hidden layer dimension
#         :param out_feats: output dimension
#         :param num_rels: number of
#         :param device: GPU or CPU
#         '''
#         super(HGNN_Model, self).__init__()
#         self.layer1 = RelGraphConv(in_feats, hidden_feats, num_rels, "basis")
#         self.layer2 = RelGraphConv(hidden_feats, out_feats, num_rels, "basis")
#
#     def forward(self, g, features,etype):
#         h = self.layer1(g, features,etype)
#         h = torch.relu(h)
#         h = self.layer2(g, h,etype)
#         return h


# heterogeneous graph neural network
class HGNN_Model(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_feats,
                 out_feats,
                 num_rels=3,
                 device='cuda'):
        '''
        :param in_feats: input dimension
        :param hidden_feats: hidden layer dimension
        :param out_feats: output dimension
        :param num_rels: number of
        :param device: GPU or CPU
        '''
        super(HGNN_Model, self).__init__()
        self.layer1 = dglnn.HeteroGraphConv({
                                'C_to_C' : dglnn.GraphConv(in_feats=in_feats,out_feats=hidden_feats),
                                'C_to_N' : dglnn.GraphConv(in_feats=in_feats,out_feats=hidden_feats),
                                'N_to_N' : dglnn.SAGEConv(in_feats=in_feats,out_feats=hidden_feats,aggregator_type='mean')},
                                aggregate='sum')

        self.layer2 = dglnn.HeteroGraphConv({
                                'C_to_C' : dglnn.GraphConv(in_feats=hidden_feats,out_feats=out_feats),
                                'C_to_N' : dglnn.GraphConv(in_feats=hidden_feats,out_feats=out_feats),
                                'N_to_N' : dglnn.SAGEConv(in_feats=hidden_feats,out_feats=out_feats,aggregator_type='mean')},
                                aggregate='sum')

    def forward(self, g, features):
        h = self.layer1(g, features)
        for key in h.keys():
            h[key] = torch.relu(h[key])
        h = self.layer2(g, h)
        return h

class MLP(torch.nn.Module):
    def __init__(self, sizes, batch_norm=True):
        """
        :param sizes: list of sizes of the layers, for example:[100,200,100]
        :param batch_norm: whether to use batch normalization
        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# multi-modal encoder
class Encoder(nn.Module):
    def __init__(self, hidden_dim=512, desc_embed_dim=768, seq_embed_dim=768):
        super(Encoder, self).__init__()

        self.desc_fc = ProjectionHead_desc(embedding_dim=desc_embed_dim,projection_dim=hidden_dim)
        self.seq_fc = ProjectionHead_seq(embedding_dim=seq_embed_dim,projection_dim=hidden_dim)
        # self.desc_fc = MLP([desc_embed_dim,2*hidden_dim,hidden_dim])
        # self.seq_fc = MLP([seq_embed_dim,2*hidden_dim,hidden_dim])
        self.fc = MLP([2*hidden_dim,hidden_dim,hidden_dim])
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu=torch.nn.ReLU()

        self.descriminator=Discriminator(hidden_dim=hidden_dim)


    # contrastive learning
    def logits(self, desc_embed, seq_embed, temp=1):

        desc_proj = self.relu(self.desc_fc(desc_embed))
        seq_proj = self.relu(self.seq_fc(seq_embed))

        sim=self.descriminator(desc_proj,seq_proj)/temp

        pos = torch.diagonal(sim)
        sim_exp = torch.exp(sim)
        sum_exp = sim_exp.sum(dim=1, keepdim=True)

        loss=-torch.log(torch.exp(pos)/sum_exp.squeeze(1))

        return torch.mean(loss)


    def forward(self, desc_embed, seq_embed):

        desc_proj = self.relu(self.desc_fc(desc_embed))
        seq_proj = self.relu(self.seq_fc(seq_embed))

        combined_features = torch.cat((desc_proj, seq_proj), dim=-1)

        output = self.fc(combined_features)

        return output


# class Discriminator(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Discriminator, self).__init__()
#         self.f_k = nn.Bilinear(hidden_dim, hidden_dim, 1)
#         for m in self.modules():
#             self.weights_init(m)
#         self.sigmoid = nn.Sigmoid()
#
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Bilinear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     def forward(self, h_a, h_b):
#         n = h_a.shape[0]
#
#         c = h_a.unsqueeze(1).expand(-1, n, -1)  # Shape: [n, n, hidden_dim]
#         sim = self.f_k(c, h_b.unsqueeze(0).expand(n, -1, -1))  # Shape: [n, n, 1]
#
#         logits = sim.squeeze(-1)  # Shape: [n, n]
#
#         return logits


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc_1 = MLP(sizes=[hidden_dim,256,128])
        self.fc_2 = MLP(sizes=[hidden_dim,256,128])

    def forward(self, f_a, f_b):
        h_a=self.fc_1(f_a)
        h_b=self.fc_2(f_b)
        norms_a=h_a.norm(p=2,dim=1,keepdim=True)
        norms_b = h_b.norm(p=2, dim=1, keepdim=True)

        logits=torch.mm(h_a/norms_a,(h_b/norms_b).T)

        return logits

class Multimodal_Encoder(nn.Module):
    def __init__(self, hidden_dim=512, desc_embed_dim=768, seq_embed_dim=768,lr=1e-3):
        super(Multimodal_Encoder, self).__init__()
        self.encoder=Encoder(hidden_dim, desc_embed_dim, seq_embed_dim)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=lr)

    def forward(self,desc_embed, seq_embed):
        return self.encoder(desc_embed,seq_embed),self.encoder.desc_fc(desc_embed),self.encoder.seq_fc(seq_embed)

    def alignment(self,desc_embed, seq_embed):
        self.optimizer.zero_grad()
        loss=self.encoder.logits(desc_embed, seq_embed)
        loss.backward()
        self.optimizer.step()

        return loss.item()


class MultiHeadSimilarity(nn.Module):
    def __init__(self, input_dim, num_heads=3, similarity_type='cosine'):
        """
        Multi-head similarity calculation module for single pair of sequences.

        :param input_dim: The dimension of input features.
        :param num_heads: The number of attention heads.
        :param similarity_type: The type of similarity metric ('cosine' or 'dot').
        """
        super(MultiHeadSimilarity, self).__init__()
        self.num_heads = num_heads
        self.similarity_type = similarity_type

        # We will use a linear layer to project the inputs to different "heads"
        self.heads = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_heads)])

    def forward(self, query):
        """
        Forward pass to compute multi-head similarity between `query` and `key`.

        :param query: A tensor of shape (n, input_dim).
        :return: A tensor of similarity scores of shape (n, n).
        """
        # Store similarity scores from each head
        head_similarities = []

        # Compute similarity for each head
        for i in range(self.num_heads):
            # Project query and key to a new space using each head
            query_proj = self.heads[i](query)  # Shape: (seq_len, input_dim)

            # Compute similarity based on chosen metric (Cosine or Dot product)
            similarity=cosine_similarity(query_proj)


            head_similarities.append(similarity)

        # Average the results from all heads (shape: seq_len, seq_len)
        avg_similarity = torch.mean(torch.stack(head_similarities, dim=0), dim=0)

        return avg_similarity


# class ChannelAttention(nn.Module):
#     def __init__(self, reduction_ratio=16):
#         super(ChannelAttention, self).__init__()
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
#         self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
#
#         self.fc1 = nn.Conv2d(1, max(1, 1 // reduction_ratio), kernel_size=1)
#         self.fc2 = nn.Conv2d(max(1, 1 // reduction_ratio), 1, kernel_size=1)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.avg_pool(x)
#         max_out = self.max_pool(x)
#
#         avg_out = self.fc2(F.relu(self.fc1(avg_out)))
#         max_out = self.fc2(F.relu(self.fc1(max_out)))
#
#         out = avg_out + max_out
#
#         attention = self.sigmoid(out)
#
#         return x * attention
#
#
# class FusionModule(nn.Module):
#     def __init__(self, reduction_ratio=16):
#         super(FusionModule, self).__init__()
#         self.channel_attention = ChannelAttention(reduction_ratio)
#
#     def forward(self, x1, x2, x3):
#         x1_att = self.channel_attention(x1.unsqueeze(0).unsqueeze(0))
#         x2_att = self.channel_attention(x2.unsqueeze(0).unsqueeze(0))
#         x3_att = self.channel_attention(x3.unsqueeze(0).unsqueeze(0))
#
#         # fused = (x1_att + x2_att + x3_att) / 3
#         fused = (x1_att + x2_att + x3_att)/3

        return fused.squeeze(0).squeeze(0)


class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # equal weight
        # if weights != None:
        #     # self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))
        #     with torch.no_grad():
        #         w = torch.Tensor(weights).reshape(self.weight.shape)
        #         self.weight.copy_(w)

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        # Row normalization of all graphs generated
        adj_list = F.normalize(adj_list, dim=1, p=1)
        # Hadamard product + summation -> Conv
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)


class ProjectionHead_desc(nn.Module):
    def __init__(
            self,
    embedding_dim,
    projection_dim
    ):
        super().__init__()
        self.projection=nn.Linear(embedding_dim,projection_dim)
        self.gelu=nn.GELU()
        self.fc=nn.Linear(projection_dim,projection_dim)
        self.norm_output=nn.LayerNorm(projection_dim)
        self.sigma=0.1


    def forward(self,x):
        x = torch.exp(-x ** 2 / (2 * (self.sigma ** 2)))
        projected=self.projection(x)
        x=self.gelu(projected)
        x=self.fc(x)
        x=torch.add(x,projected)
        x=self.norm_output(x)
        return x


class ProjectionHead_seq(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim
            ):
        super().__init__()
        self.projection=nn.Linear(embedding_dim,projection_dim)
        self.gelu=nn.GELU()
        self.fc=nn.Linear(projection_dim,projection_dim)
        self.norm_output=nn.LayerNorm(projection_dim)
        self.sigma=0.1


    def forward(self,x):
        x = torch.exp(-x ** 2 / (2 * (self.sigma ** 2)))
        projected=self.projection(x)
        x=self.gelu(projected)
        x=self.fc(x)
        x=torch.add(x,projected)
        x=self.norm_output(x)
        return x

# class ProjectionHead_seq(nn.Module):
#     def __init__(
#             self,
#             embedding_dim,
#             projection_dim,
#             sigma=0.01
#     ):
#         super().__init__()
#         self.projection = nn.Linear(embedding_dim, projection_dim)
#         self.gelu = nn.GELU()
#         self.fc = nn.Linear(projection_dim, projection_dim)
#         self.norm_input=nn.LayerNorm(embedding_dim)
#         self.norm_output = nn.LayerNorm(projection_dim)
#         self.sigma=sigma
#
#     def forward(self, x):
#         x=torch.exp(-x**2/(2*(self.sigma**2)))
#         x=self.norm_input(x)
#         projected = self.projection(x)
#         x = self.gelu(projected)
#         x = self.fc(x)
#         x = torch.add(x, projected)
#         x = self.norm_output(x)
#         return x


class MultiLayerGCN(nn.Module):
    def __init__(self, feats, num_layers, activation=F.relu):
        super(MultiLayerGCN, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        self.layers = nn.ModuleList()


        for i in range(num_layers):
            self.layers.append(GraphConv(feats[i], feats[i+1]))

        # self.layers.append(GraphConv(feats[-2], feats[-1]))


    # return a list
    def forward(self, A, features,threshold=0.04):
        with torch.no_grad():
            A=torch.where(A >= threshold, A, torch.tensor(0.0))

        row, col = torch.nonzero(A, as_tuple=True)
        g = dgl.graph((row, col), num_nodes=A.shape[0])
        nodes=torch.arange(A.shape[0]).to('cuda')
        g.add_edges(nodes,nodes)

        layer_outputs = []
        h = features
        for i in range(self.num_layers):
            h = self.layers[i](g, h)
            h = self.activation(h)
            layer_outputs.append(h)

        return layer_outputs

class multiLayerGAT(nn.Module):
    def __init__(self, dim=[256,256,64], num_heads=8,feat_drop=0.4,attn_drop=0.4):
        super(multiLayerGAT, self).__init__()
        self.num_heads=num_heads
        self.gat_layers=nn.ModuleList()
        self.gat_layers.append(
            GATConv(dim[0], dim[1], num_heads=num_heads, feat_drop=feat_drop, attn_drop=attn_drop,allow_zero_in_degree=True))
        for i in range(len(dim)-2):
            self.gat_layers.append(GATConv(num_heads*dim[i+1],dim[i+2],num_heads=num_heads,feat_drop=feat_drop,attn_drop=attn_drop,allow_zero_in_degree=True))

    def forward(self,g,feature):
        h=feature
        for layer in self.gat_layers:
            # h=F.relu(layer(g,h.reshape(h.shape[0],-1)))
            h = F.relu(layer(g, h.flatten(1)))
        # [num_gene,head_num,d]->[num_gene,d]
        return torch.mean(h,dim=1)




class HeterGAT(nn.Module):
    def __init__(self, layer_dim=[256,256,64],num_heads=8,edge_type=['CC','CN','NN'], dropout_ft=0.4,dropout_att=0.4):
        super(HeterGAT, self).__init__()

        self.heter_gat=nn.ModuleList()
        self.edge_type=edge_type


        for _ in range(len(edge_type)):
            self.heter_gat.append(multiLayerGAT(dim=layer_dim,num_heads=num_heads,feat_drop=dropout_ft,attn_drop=dropout_att))

    def forward(self,sub_graph,features,aggregation='sum'):
        f_list=[]
        for i in range(len(sub_graph)):
            with torch.no_grad():
                g=dgl.graph(torch.where(sub_graph[i]>0),num_nodes=sub_graph[i].shape[0])

            f=self.heter_gat[i](g,features)
            f_list.append(f)


        if aggregation=='sum':
            output=torch.stack(f_list,dim=0)
            output=torch.sum(output,dim=0)

        return output