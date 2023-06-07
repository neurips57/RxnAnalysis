import numpy as np
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

import dgl
from dgl.nn.pytorch import NNConv, Set2Set

from util import MC_dropout
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class MPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, hidden_feats = 64 ,
                 num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 1025):
        
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
        
        self.gnn_layer = NNConv(
            in_feats = hidden_feats,
            out_feats = hidden_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.activation = nn.ReLU()
        
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        self.readout = Set2Set(input_dim = hidden_feats * 2,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )
             
    def forward(self, g):
            
        node_feats = g.ndata['attr']
        edge_feats = g.edata['edge_attr']
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)
        node_aggr = [node_feats]        
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(g, node_aggr)
        graph_feats = self.sparsify(readout)
        
        return graph_feats

class reactionMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, n_rmol, n_imol,
                 readout_feats = 1025,
                 predict_hidden_feats = 512, prob_dropout = 0.1, cuda = torch.device('cuda:0')):
        
        super(reactionMPNN, self).__init__()
        self.n_rmol = n_rmol
        self.n_imol = n_imol
        self.readout_feats = readout_feats
        self.mpnn = MPNN(node_in_feats, edge_in_feats)
        self.one_hot_len = n_rmol + n_imol + 1
        self.d_model = self.readout_feats + self.one_hot_len
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = 8).to(cuda)

        self.predict = [nn.Sequential(
            nn.Linear((readout_feats + n_rmol + n_imol+1) * i, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout))
            .to(cuda) for i in range(n_rmol + 1, n_rmol + n_imol + 2)]
    
        self.predict2 = nn.Sequential(
            nn.Linear(predict_hidden_feats * (n_imol), predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout))

        self.predict3 = nn.Sequential(
            nn.Linear(predict_hidden_feats, 2)
        )

    def forward(self, rmols, imols, pmols):
        
        cuda = torch.device('cuda:0')
        batch_sz = self.mpnn(rmols[0]).size(0)
        one_hot_len = self.n_rmol + self.n_imol + 1
        d_model = self.readout_feats + one_hot_len
        
        r_perm = list(range(self.n_rmol))
        r_graph_feats = []
        for i in range(self.n_rmol):
            one_hot = [[r_perm[i] == j for j in range(one_hot_len)] for _ in range(batch_sz)]
            one_hot_tensor = torch.Tensor(one_hot).to(cuda)
            r_graph = torch.reshape(torch.cat([self.mpnn(rmols[i]), one_hot_tensor], 1),(1,batch_sz ,d_model))
            r_graph_feats.append(r_graph)
  
        r_graph_feats_tensor = torch.cat(r_graph_feats,0)

        i_graph_feats = []
        for i in range(self.n_imol):
            one_hot = [[i + self.n_rmol == j for j in range(one_hot_len)] for _ in range(batch_sz)]
            one_hot_tensor = torch.Tensor(one_hot).to(cuda)
            i_graph = torch.reshape(torch.cat([self.mpnn(imols[i]), one_hot_tensor], 1),(1,batch_sz ,d_model))
            i_graph_feats.append(i_graph)

        i_graph_feats_tensor = torch.cat(i_graph_feats,0)

        p_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in pmols]), 0)
        p_graph_feats_tensor = torch.cat([p_graph_feats, torch.Tensor([[i==(one_hot_len-1) for i in range(one_hot_len)] for _ in range(batch_sz)]).to(cuda)], 1)
        p_graph_feats_tensor = torch.reshape(p_graph_feats_tensor,(1,batch_sz ,d_model))

        '''
        torch.nn.transformer - define transformer layer
        input: (seq, batch, feature) = (10, batch_sz , 1034)
        dmodel = 1024 + 10 = 1034
        '''
        concat_feats = torch.cat([r_graph_feats_tensor, i_graph_feats_tensor, p_graph_feats_tensor], 0)
        out = self.encoder_layer(concat_feats)

        '''
        send r + i1 + p, r + i1 + i2 + p, r + i1 + i2 + i3 + p through NN
        '''
        r = [torch.reshape(out[i],(batch_sz ,d_model)) for i in range(self.n_rmol)]
        it = [torch.reshape(out[i],(batch_sz ,d_model)) for i in range(self.n_rmol,self.n_rmol+self.n_imol)]
        p = torch.reshape(out[self.n_rmol+self.n_imol],(batch_sz ,d_model)) 
        it_p = it+[p]
        delta_e = []
        for i in range(2,self.n_imol+2):
            feats = torch.cat([torch.cat(r, 1), torch.cat(it_p[0:i], 1)], 1)
            del_e = self.predict[i-1](feats)
            delta_e.append(del_e)

        del_ee = torch.cat(delta_e, 1)
        
        '''
        send through a FNN - predict yield
        '''
        pre = self.predict2(del_ee)
        res = self.predict3(pre)   
        return res[:,0], res[:,1]

def training(net, out_file, train_loader, val_loader, test_loader, train_y_mean, train_y_std, model_path, val_monitor_epoch = 1, n_forward_pass = 5, cuda = torch.device('cuda:0')):

    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size
    
    val_mae = []
    test_result = []
    
    print('train size: ', train_size)
    print('train size: ', train_size, file=out_file)
    
    try:
        rmol_max_cnt = train_loader.dataset.dataset.rmol_max_cnt
        imol_max_cnt = train_loader.dataset.dataset.imol_max_cnt
        pmol_max_cnt = train_loader.dataset.dataset.pmol_max_cnt
    except:
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt
        imol_max_cnt = train_loader.dataset.imol_max_cnt
        pmol_max_cnt = train_loader.dataset.pmol_max_cnt

    loss_fn = nn.MSELoss(reduction = 'none')

    n_epochs = 500
    optimizer = Adam(net.parameters(), lr = 1e-4, weight_decay = 1e-5)
    lr_scheduler = MultiStepLR(optimizer, milestones = [400, 450], gamma = 0.1, verbose = False)

    for epoch in range(n_epochs):
        
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(train_loader):
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_imol = [b.to(cuda) for b in batchdata[rmol_max_cnt:rmol_max_cnt+imol_max_cnt]]
            inputs_pmol = [b.to(cuda) for b in batchdata[rmol_max_cnt+imol_max_cnt:rmol_max_cnt+imol_max_cnt+pmol_max_cnt]]
            
            labels = (batchdata[-1] - train_y_mean) / train_y_std
            labels = labels.to(cuda)
            
            
            pred, logvar = net(inputs_rmol, inputs_imol, inputs_pmol)

            loss = loss_fn(pred, labels)

            pred_ = pred*train_y_std + train_y_mean
            labels_ = labels*train_y_std + train_y_mean
            loss1 = loss_fn(pred_, labels_)
            loss1 = loss1.mean()

            loss = (1) * loss.sum() + 0 * ( loss * torch.exp(-logvar) + logvar ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = loss.detach().item()
            train_loss1 = loss1.detach().item()
            
        model_path1 = model_path + f'{epoch}.pt'
        torch.save(net.state_dict(), model_path1)  
             
        print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f'
              %(epoch, optimizer.param_groups[-1]['lr'], train_size, train_size, train_loss1**0.5, (time.time()-start_time)/60))
        print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f'
              %(epoch, optimizer.param_groups[-1]['lr'], train_size, train_size, train_loss1**0.5, (time.time()-start_time)/60), file=out_file)    
        
        lr_scheduler.step()

        # validation
        if val_loader is not None and (epoch + 1) % val_monitor_epoch == 0:
            
            val_y = val_loader.dataset.dataset.yld[val_loader.dataset.indices]
            val_y_pred, _, _ = inference(net, val_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)

            result = [mean_absolute_error(val_y, val_y_pred),
                      mean_squared_error(val_y, val_y_pred) ** 0.5,
                      r2_score(val_y, val_y_pred)]

            val_mae.append(result[0])
                      
            print('--- validation at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f' %(epoch, len(val_y), result[0], result[1], result[2]))
            print('--- validation at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f' %(epoch, len(val_y), result[0], result[1], result[2]), file=out_file)


        if test_loader is not None and (epoch + 1) % val_monitor_epoch == 0:
            
            test_y = test_loader.dataset.dataset.yld[test_loader.dataset.indices]
            test_y_pred, _, _ = inference(net, test_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)

            result = [mean_absolute_error(test_y, test_y_pred),
                      mean_squared_error(test_y, test_y_pred) ** 0.5,
                      r2_score(test_y, test_y_pred)]

            test_result.append(result)
                      
            print('--- test at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f' %(epoch, len(test_y), result[0], result[1], result[2]))
            print('--- test at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f' %(epoch, len(test_y), result[0], result[1], result[2]), file=out_file)

    print('training terminated at epoch %d' %epoch)
    print('training terminated at epoch %d' %epoch, file=out_file)
    idx = np.argmin(np.array(val_mae))
    result = test_result[idx]
    print('--- test at MIN validation at epoch %d MAE %.3f RMSE %.3f R2 %.3f' %(idx, result[0], result[1], result[2]))
    print('--- test at MIN validation at epoch %d MAE %.3f RMSE %.3f R2 %.3f' %(idx, result[0], result[1], result[2]), file=out_file)
    
    return net
    

def inference(net, test_loader, train_y_mean, train_y_std, n_forward_pass = 30, cuda = torch.device('cuda:0')):

    batch_size = test_loader.batch_size

    
    try:
        rmol_max_cnt = test_loader.dataset.dataset.rmol_max_cnt
        imol_max_cnt = test_loader.dataset.dataset.imol_max_cnt
        pmol_max_cnt = test_loader.dataset.dataset.pmol_max_cnt
    except:
        rmol_max_cnt = test_loader.dataset.rmol_max_cnt
        imol_max_cnt = test_loader.dataset.imol_max_cnt
        pmol_max_cnt = test_loader.dataset.pmol_max_cnt
             
    net.eval()
    MC_dropout(net)
    
    test_y_mean = []
    test_y_var = []
    
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
        
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_imol = [b.to(cuda) for b in batchdata[rmol_max_cnt:rmol_max_cnt+imol_max_cnt]]
            inputs_pmol = [b.to(cuda) for b in batchdata[rmol_max_cnt+imol_max_cnt:rmol_max_cnt+imol_max_cnt+pmol_max_cnt]]
            
            mean_list = []
            var_list = []
            
            for _ in range(n_forward_pass):
                mean, logvar = net(inputs_rmol, inputs_imol, inputs_pmol)
                
                mean_list.append(mean.cpu().numpy())
                var_list.append(np.exp(logvar.cpu().numpy()))

            test_y_mean.append(np.array(mean_list).transpose())
            test_y_var.append(np.array(var_list).transpose())

    test_y_mean = np.vstack(test_y_mean) * train_y_std + train_y_mean
    test_y_var = np.vstack(test_y_var) * train_y_std ** 2
    
    test_y_pred = np.mean(test_y_mean, 1)
    test_y_epistemic = np.var(test_y_mean, 1)
    test_y_aleatoric = np.mean(test_y_var, 1)
    
    return test_y_pred, test_y_epistemic, test_y_aleatoric
