'''
test api
'''
import torch.mps
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pertdata import *
from utils import construct_GRN
from modules import *
from tqdm import tqdm
from model import *
import pickle
from pytorch_lightning import Trainer
# torch.cuda.set_per_process_memory_fraction(0.8)


os.environ["passing CUDA_LAUNCH_BLOCKING"]="1"

data_name="adamson"
# data_name="norman"
hvg_num=5000
pertdata=PertData(data_name=data_name,hvg_num=hvg_num)
model=model(pertdata).to('cuda')
from pytorch_lightning.strategies import DDPStrategy


train_dataset = AnnDataBatchDataset(pertdata.training_cell)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=8)

torch.cuda.empty_cache()
model.train_encoder()
# torch.autograd.set_detect_anomaly(True)

trainer=Trainer(max_epochs=3,devices=3,accelerator="gpu",strategy=DDPStrategy(find_unused_parameters=True))
trainer.fit(model,train_loader)


# optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# for epoch in range(4):
#     for batch_data, batch_condition in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
#         # optimizer.zero_grad()
#         # cond,idx=extract_cond(pertdata,batch_condition)
#         # loss=model(pertdata,batch_data,idx)
#         # loss.backward()
#         # optimizer.step()
#
#         optimizer.zero_grad()
#         # cond,idx=extract_cond(pertdata,batch_condition)
#         loss=model.training_step([batch_data,batch_condition])
#         loss.backward()
#         optimizer.step()


torch.save(model.state_dict(),"result/"+data_name+"_"+str(hvg_num)+"/model.pt")
