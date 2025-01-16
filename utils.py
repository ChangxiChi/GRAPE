import torch
import dgl
import numpy as np
import torch.nn.functional as F

# construct gene regulation network
# def construct_GRN(pert_data,
#                   sim_matrix,
#                   threshold=0.9):
#     '''
#     embed_dict: contain fused embedding of desc and seq
#     sim_matrix:
#     '''
#     gene_name=pert_data.gene_name
#     gene_biotype=pert_data.get_biotype(gene_name)
#
#     # construct heterogeneous graph
#     g = dgl.heterograph({
#         ('C', 'C-C', 'C'): [],
#         ('C', 'C-N', 'N'): [],
#         ('N', 'N-N', 'N'): []
#     })
#
#     c_nodes = []
#     n_nodes = []
#
#     for idx in range(len(gene_biotype)):
#         if gene_biotype[idx] == 'coding':
#             c_nodes.append(idx)
#         else:
#             n_nodes.append(idx)
#
#     # add coding gene
#     g.add_nodes(len(c_nodes), ntype='C')
#     # add non-coding gene
#     g.add_nodes(len(n_nodes), ntype='N')
#
#     g.nodes['N'].data['idx'] = torch.tensor(n_nodes)
#     g.nodes['C'].data['idx'] = torch.tensor(c_nodes)
#
#     added_edges = set()
#     indices = torch.where(sim_matrix > threshold)
#     src_indices=indices[0] # row
#     dst_indices=indices[1] # col
#
#     edge_weights = []
#     # record [non-coding,non-coding] edge
#     src_list_NN = []
#     dst_list_NN = []
#     # record [coding,non-coding] or [non-coding,coding] edge
#     src_list_CN = []
#     dst_list_CN = []
#     # record [coding,coding] edge
#     src_list_CC = []
#     dst_list_CC = []
#
#     edge_types=[]
#
#     for src, dst in zip(src_indices, dst_indices):
#         # indirection check (avoid adding duplicate edges)
#         if (src.item(), dst.item()) in added_edges or (dst.item(), src.item()) in added_edges:
#             continue
#
#         if gene_biotype[src] == 'coding' and gene_biotype[dst] == 'coding' :
#             edge_type = 'C-C'
#             src_list_CC.append(src.item())
#             dst_list_CC.append(dst.item())
#
#         elif gene_biotype[src] == 'non-coding' and gene_biotype[dst] == 'non-coding' :
#             edge_type = 'N-N'
#             src_list_NN.append(src.item()-len(c_nodes))
#             dst_list_NN.append(dst.item()-len(c_nodes))
#
#         elif gene_biotype[src] == 'coding' and gene_biotype[dst] == 'non-coding':
#             edge_type = 'C-N'
#             src_list_CN.append(src.item())
#             dst_list_CN.append(dst.item()-len(c_nodes))
#
#         edge_types.append(edge_type)
#         edge_weights.append(sim_matrix[src.item(), dst.item()])
#
#         added_edges.add((src.item(), dst.item()))
#
#     # Add edges to the graph
#     src_list_CC=torch.tensor(src_list_CC,dtype=torch.int64)
#     dst_list_CC=torch.tensor(dst_list_CC,dtype=torch.int64)
#     src_list_CN=torch.tensor(src_list_CN,dtype=torch.int64)
#     dst_list_CN=torch.tensor(dst_list_CN,dtype=torch.int64)
#     src_list_NN=torch.tensor(src_list_NN,dtype=torch.int64)
#     dst_list_NN=torch.tensor(dst_list_NN,dtype=torch.int64)
#
#     g.add_edges(src_list_CC, dst_list_CC, etype='C-C')
#     g.add_edges(src_list_CN, dst_list_CN, etype='C-N')
#     g.add_edges(src_list_NN, dst_list_NN, etype='N-N')
#
#     # Assign edge weights based on edge type
#     for edge_type in ['C-C', 'C-N', 'N-N']:
#         edge_mask = [i for i, etype in enumerate(edge_types) if etype == edge_type]
#
#         # Filter edge weights for the current edge type
#         type_edge_weights = torch.tensor(edge_weights)[edge_mask]
#
#         # Assign weights to corresponding edges in the graph
#         g.edges[edge_type].data['weight'] = type_edge_weights
#
#     return g,edge_types

# return perturbation gene name
# and gene index
def extract_cond(pert_data,batch_condition):
    cond=[]
    idx=[]
    for condition in batch_condition:
        if condition=='ctrl':
            cond.append(condition)
            idx.append(-1)
        else:
            condition = [part for part in condition.split('+') if part != "ctrl"][0]
            idx.append(pert_data.get_gene_idx([condition])[0])
            cond.append(condition)

    return cond,idx


# def construct_GRN(pert_data,
#                   sim_matrix,
#                   threshold=0.1):
#     '''
#     embed_dict: contain fused embedding of desc and seq
#     sim_matrix:
#     '''
#     c_num=pert_data.c_num
#
#     # construct heterogeneous graph
#     sim_matrix=F.normalize(sim_matrix)
#
#     indices = torch.where(sim_matrix > threshold)
#
#
#     src_indices=indices[0] # row
#     dst_indices=indices[1] # col
#
#
#     g=dgl.graph((src_indices,dst_indices),num_nodes=pert_data.gene_num)
#
#
#     # C-C: 0
#     # N-N: 1
#     # C-N: 2
#
#     edge_types = np.zeros(len(indices[0]), dtype=int)
#     mask_0 = (np.array(indices[0].cpu()) < c_num) & (np.array(indices[1].cpu()) < c_num)
#     mask_1 = (np.array(indices[0].cpu()) >= c_num) & (np.array(indices[1].cpu()) >= c_num)
#     mask_2 = ~(mask_0 | mask_1)
#
#     edge_types[mask_0] = 0
#     edge_types[mask_1] = 1
#     edge_types[mask_2] = 2
#
#
#     return g,torch.tensor(edge_types).to('cuda')


def construct_GRN(pert_data,
                  sim_matrix,
                  threshold=0.1):
    '''
    embed_dict: contain fused embedding of desc and seq
    sim_matrix:
    '''
    c_num = pert_data.c_num

    # construct heterogeneous graph
    sim_matrix=torch.where(sim_matrix > threshold, sim_matrix, torch.tensor(0.0))

    u, v = torch.where(sim_matrix)
    u, v = u.squeeze(), v.squeeze()

    c_to_c_mask = (u < c_num) & (v < c_num)
    c_to_c_u = u[c_to_c_mask]
    c_to_c_v = v[c_to_c_mask]

    c_to_n_mask = (u < c_num) & (v >= c_num)
    c_to_n_u = u[c_to_n_mask]
    c_to_n_v = v[c_to_n_mask]-c_num

    n_to_n_mask = (u >= c_num) & (v >= c_num)
    n_to_n_u = u[n_to_n_mask]-c_num
    n_to_n_v = v[n_to_n_mask]-c_num

    g = dgl.heterograph({
        ('C', 'C_to_C', 'C'): (c_to_c_u, c_to_c_v),
        ('C', 'C_to_N', 'N'): (c_to_n_u, c_to_n_v),
        ('N', 'N_to_N', 'N'): (n_to_n_u, n_to_n_v),
    })


    return g


def cosine_similarity(features, epsilon=1e-8):
    norms = torch.norm(features, p=2, dim=1, keepdim=True)

    norms = torch.clamp(norms, min=epsilon)
    similarity_matrix = torch.mm(features, features.t())
    similarity_matrix = similarity_matrix / (norms * norms.t())

    return similarity_matrix