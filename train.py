import numpy as np
import torch.nn as nn
import os
import copy
from tqdm import trange
import pandas as pd
from torch_geometric.data import DataLoader
from utils import *
from MGN import *

def train(dataset, device, stats_list, args):
    df = pd.DataFrame(columns=['epoch','train_loss','test_loss', 'velo_val_loss'])

    #Define the model name for saving 
    model_name='model_nl'+str(args.num_layers)+'_bs'+str(args.batch_size) + \
               '_hd'+str(args.hidden_dim)+'_ep'+str(args.epochs)+'_wd'+str(args.weight_decay) + \
               '_lr'+str(args.lr)+'_shuff_'+str(args.shuffle)+'_tr'+str(args.train_size)+'_te'+str(args.test_size)
    
    #load data
    loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[args.train_size], batch_size=args.batch_size, shuffle=False)

    #collect statistical information
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
        std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))
    
    #build model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 2
    model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes, args).to(device)
    scheduler, optimizer = build_optimizer(args, model.parameters())

    #train
    print('Training...')
    losses = []
    test_losses = []
    velocity_val = []
    best_test_loss = np.inf
    best_model = None
    model.train()
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        model.train()
        loss_all = 0
        count = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss = model.loss(pred, data, mean_vec_y, std_vec_y)
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
            count += 1
        loss = loss_all / count
        losses.append(loss)
        if epoch % 10 == 0:
            if args.save_velo_val:
                test_loss, vel_val = test(model, loader, device, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, args)
                velocity_val.append(vel_val.item())
            else:
                test_loss, _ = test(model, loader, optimizer, scheduler, device, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, args)
            test_losses.append(test_loss)
            # saving model
            if not os.path.isdir( args.checkpoint_dir ):
                os.mkdir(args.checkpoint_dir)
            PATH = os.path.join(args.checkpoint_dir, model_name+'.csv')
            df.to_csv(PATH,index=False)
            #save the model if the current one is better than the previous best
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)
        else:
            if(args.save_velo_val):
                test_losses.append(test_losses[-1])
                velocity_val.append(velocity_val[-1])
        #if (args.save_velo_val):
            #df = df.append({'epoch': epoch,'train_loss': losses[-1],
        #                    'test_loss':test_losses[-1],
        #                   'velo_val_loss': velocity_val[-1]}, ignore_index=True)
        #else:
            #df = df.append({'epoch': epoch, 'train_loss': losses[-1], 'test_loss': test_losses[-1]}, ignore_index=True)
        if epoch % 100 == 0:
            if (args.save_velo_val):
                print("train loss", str(round(loss_all, 2)),
                      "test loss", str(round(test_loss, 2)),
                      "velo loss", str(round(velocity_val[-1], 5)))
            else:
                print("train loss", str(round(loss_all,2)), "test loss", str(round(test_loss[-1],2)))
            if(args.save_best_model):
                PATH = os.path.join(args.checkpoint_dir, model_name+'.pt')
                torch.save(best_model.state_dict(), PATH)
    return test_losses, losses, velocity_val, best_model, best_test_loss, test_loader


def test(model, loader, device, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, is_validation,
          delta_t=0.01, save_model_preds=False, model_type=None):
    model.eval()
    loss_all = 0
    count = 0
    velo_val = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss = model.loss(pred, data, mean_vec_y, std_vec_y)
            loss_all += loss.item()
            count += 1
            if (is_validation):
                normal = torch.tensor(0)
                outflow = torch.tensor(5)
                loss_mask = torch.logical_or((torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(0)),
                                             (torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(5)))

                eval_velo = data.x[:, 0:2] + unnormalize( pred[:], mean_vec_y, std_vec_y ) * delta_t
                gs_velo = data.x[:, 0:2] + data.y[:] * delta_t
                
                error = torch.sum((eval_velo - gs_velo) ** 2, axis=1)
                velo_val += torch.sqrt(torch.mean(error[loss_mask]))
    return loss_all / count, velo_val / count