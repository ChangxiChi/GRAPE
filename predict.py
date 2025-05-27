import numpy as np

from pertdata import *
from model import *
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance


data_name='adamson'
hvg_num=5000

pertdata=PertData(data_name=data_name,hvg_num=hvg_num)
model=model(pertdata).cuda()
model.load_state_dict(torch.load("./result/"+data_name+"_"+str(hvg_num)+"/model.pt",weights_only=True))

model.eval()

sum_1 = []  # RMSE
p_1 = []  # pearson correlation coefficient
p_2 = []  # pearson DE20
p_3 = []
pcc=[]
dir_acc = []
dir_de_20 = []
dir_de_40 = []


with torch.no_grad():
    for perturb in pertdata.testing_cond:

        if data_name == 'adamson':
            cell = pertdata.adata[pertdata.adata.obs['condition'].isin([perturb + "+ctrl"]), :]
        elif data_name == 'norman':
            cell = pertdata.adata[pertdata.adata.obs['condition'].isin([perturb + "+ctrl", "ctrl+" + perturb]), :]

        ctrl = pertdata.adata[pertdata.adata.obs['condition'].isin(['ctrl']), :]
        cond, idx = extract_cond(pertdata, [perturb])
        pred = model.prediction(idx).cpu()
        pred = pred.numpy()

        # perturbation key of DE dict
        if data_name == "adamson":
            de_key = 'K562(?)_' + perturb + '+ctrl_1+1'
        elif data_name == 'norman':
            if 'A549_' + perturb + '+ctrl_1+1' in pertdata.adata.uns['rank_genes_groups_cov_all'].keys():
                de_key = 'A549_' + perturb + '+ctrl_1+1'
            else:
                de_key = 'A549_ctrl+' + perturb + '_1+1'

        id_list = pertdata.adata.uns['rank_genes_groups_cov_all'][de_key]
        gene_name = pertdata.get_id_gene(list(id_list))

        gene_name_20 = [gene for gene in gene_name if gene in pertdata.gene_name][:20]
        gene_name_40 = [gene for gene in gene_name if gene in pertdata.gene_name][:40]

        gene_index_20 = pertdata.get_gene_idx(gene_name_20)
        gene_index_40 = pertdata.get_gene_idx(gene_name_40)

        mean_pre = np.mean(ctrl.X.toarray(), axis=0)
        mean_post = np.mean(cell.X.toarray(), axis=0)


        cell_type=cell.obs['cell_type'].unique().tolist()[0]
        DE=pertdata.adata.uns['rank_genes_groups_cov_all']
        gene_ID=pertdata.adata.var.index.tolist()

        # map id to gene name
        GT = cell.X.toarray()

        control = np.mean(ctrl.X.toarray(), axis=0)
        true_direction = np.sign(np.mean(GT, axis=0) - control)

        pred_direction = np.sign(pred - control)

        accuracy = (pred_direction == true_direction)
        accuracy = np.mean(accuracy)
        dir_acc.append(accuracy)

        # top k DE gene MSE
        pred_de_20 = pred[:, gene_index_20]
        GT_de_20 = GT[:, gene_index_20]

        pred_de_40 = pred[:, gene_index_40]
        GT_de_40 = GT[:, gene_index_40]

        # calculate RMSE
        mse = np.mean((pred - np.mean(GT, axis=0)) ** 2)
        sum_1.append(np.sqrt(mse))

        # calcualte DE20 dir acc
        control_de_20 = np.mean(ctrl.X.toarray(), axis=0)[gene_index_20]
        true_direction = np.sign(np.mean(GT_de_20, axis=0) - control_de_20)

        pred_direction = np.sign(pred_de_20 - control_de_20)

        accuracy = (pred_direction == true_direction)
        accuracy = np.mean(accuracy)
        dir_de_20.append(accuracy)

        # calcualte DE20 dir acc
        control_de_40 = np.mean(ctrl.X.toarray(), axis=0)[gene_index_40]
        true_direction = np.sign(np.mean(GT_de_40, axis=0) - control_de_40)

        pred_direction = np.sign(pred_de_40 - control_de_40)

        accuracy = (pred_direction == true_direction)
        accuracy = np.mean(accuracy)
        dir_de_40.append(accuracy)

        corr, _ = pearsonr(pred.squeeze(), np.mean(GT, axis=0))
        pcc.append(corr)

        corr, _ = pearsonr(pred.squeeze() - control, np.mean(GT, axis=0) - control)
        p_1.append(corr)

        corr, _ = pearsonr(pred_de_20.squeeze() - control_de_20, np.mean(GT_de_20, axis=0) - control_de_20)
        p_2.append(corr)

        corr, _ = pearsonr(pred_de_40.squeeze() - control_de_40, np.mean(GT_de_40, axis=0) - control_de_40)
        p_3.append(corr)


print("RMSE: ", np.mean(sum_1), " variencr", np.var(sum_1))
print("aver Dir Acc: ", np.mean(dir_acc), " variencr", np.var(dir_acc))
print("DE20 Dir Acc: ", np.mean(dir_de_20), " variencr", np.var(dir_de_20))
print("DE40 Dir Acc: ", np.mean(dir_de_40), " variencr", np.var(dir_de_40))
print("Pearson_delta: ", np.mean(p_1), " variencr", np.var(p_1))
print("PCC: ", np.mean(pcc), " variencr", np.var(pcc))
print("Pearson DE20: ", np.mean(p_2), " variencr", np.var(p_2))
print("Pearson DE40: ", np.mean(p_3), " variencr", np.var(p_3))