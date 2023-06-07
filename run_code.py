import numpy as np
import sys, csv, os
import random
import torch
from torch.utils.data import DataLoader

import dgl
from dgl.data.utils import split_dataset

from dataset import GraphDataset
from util import collate_reaction_graphs
from model_df import *


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats

cuda = torch.device('cuda:0')

def setup_seed(seed): 
    random.seed(seed)                        
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  
    dgl.seed(seed)
    dgl.random.seed(seed)

setup_seed(0)

data_id = 4 #data_id 1: BH, #data_id 4: DF, #data_id 5: NS, #data_id 6: SC, #data_id 5: CM,
split_id = 0 #0-9
train_size = 518 #data_id 1: [2767], data_id 4: [518], data_id 5: [719], data_id 6: [337], data_id 7: [280]
batch_size = 16    #32
val_size = 74  #data_id 1: [395], data_id 4: [74], data_id 5: [103], data_id 6: [48], data_id 7: [40]
use_saved = False

out_file  = open('./results/results_%d_%d_%d.txt' %(data_id, split_id, train_size), 'w')
model_path = './checkpoints/model_%d_%d_%d/' %(data_id, split_id, train_size)
if not os.path.exists(model_path): os.makedirs(model_path)
            
data = GraphDataset(data_id, split_id)
train_frac_split = (train_size + 1e-5)/len(data)
val_frac_split = (val_size + 1e-5)/len(data)
train_set, val_set, test_set = split_dataset(data, [train_frac_split, val_frac_split, 1 - train_frac_split - val_frac_split], shuffle = False)
    
train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([batch_size, len(train_set)])), shuffle=False, collate_fn=collate_reaction_graphs, drop_last=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

print('-- CONFIGURATIONS')    
print('--- data_type:', data_id, split_id)
print('--- train/test: %d/%d' %(len(train_set), len(test_set)))
print('--- max no. reactants:', data.rmol_max_cnt)
print('--- max no. intermediates:', data.imol_max_cnt)
print('--- max no. products:', data.pmol_max_cnt)
print('--- use_saved:', use_saved)
print('--- model_path:', model_path)

print('-- CONFIGURATIONS', file=out_file)
print('--- data_type:', data_id, split_id)
print('--- train/test: %d/%d' %(len(train_set), len(test_set)), file=out_file)
print('--- max no. reactants:', data.rmol_max_cnt, file=out_file)
print('--- max no. intermediates:', data.imol_max_cnt, file=out_file)
print('--- max no. products:', data.pmol_max_cnt, file=out_file)
print('--- use_saved:', use_saved, file=out_file)
print('--- model_path:', model_path, file=out_file)


# training 
train_y = train_loader.dataset.dataset.yld[train_loader.dataset.indices]
train_y_mean = np.mean(train_y)
train_y_std = np.std(train_y)

node_dim = data.rmol_node_attr[0].shape[1]
edge_dim = data.rmol_edge_attr[0].shape[1]
net = reactionMPNN(node_dim, edge_dim, data.rmol_max_cnt, data.imol_max_cnt).to(cuda)

if use_saved == False:
    print('-- TRAINING')
    print('-- TRAINING', file=out_file)
    net = training(net, out_file, train_loader, val_loader, test_loader, train_y_mean, train_y_std, model_path)
    #torch.save(net.state_dict(), model_path)
else:
    print('-- LOAD SAVED MODEL')
    print('-- LOAD SAVED MODEL', file=out_file)
    net.load_state_dict(torch.load(model_path))


# inference
test_y = test_loader.dataset.dataset.yld[test_loader.dataset.indices]

test_y_pred, test_y_epistemic, test_y_aleatoric = inference(net, test_loader, train_y_mean, train_y_std)
test_y_pred = np.clip(test_y_pred, 0, 100)

result = [mean_absolute_error(test_y, test_y_pred),
        mean_squared_error(test_y, test_y_pred) ** 0.5,
        r2_score(test_y, test_y_pred),
        stats.spearmanr(np.abs(test_y-test_y_pred), test_y_aleatoric+test_y_epistemic)[0]]
            
print('-- RESULT')
print('--- test size: %d' %(len(test_y)))
print('--- MAE: %.3f, RMSE: %.3f, R2: %.3f, Spearman: %.3f' %(result[0], result[1], result[2], result[3]))

print('-- RESULT', file=out_file)
print('--- test size: %d' %(len(test_y)), file=out_file)
print('--- MAE: %.3f, RMSE: %.3f, R2: %.3f, Spearman: %.3f' %(result[0], result[1], result[2], result[3]), file=out_file)

