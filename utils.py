import enum
import torch.optim as optim
import torch

def normalize(to_vec, mean_vec, std_vec):
    return (to_vec - mean_vec) / std_vec

def unnormalize(to_vec, mean_vec, std_vec):
    return to_vec * std_vec + mean_vec

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer

def get_stats(dataset):
    mean_vec_x = torch.zeros(dataset[0].x.shape[1:])
    std_vec_x = torch.zeros(dataset[0].x.shape[1:])
    mean_vec_edge = torch.zeros(dataset[0].edge_attr.shape[1:])
    std_vec_edge = torch.zeros(dataset[0].edge_attr.shape[1:])
    mean_vec_y = torch.zeros(dataset[0].y.shape[1:])
    std_vec_y = torch.zeros(dataset[0].y.shape[1:])
    
    max_count = 10**6

    count_x = 0
    count_edge = 0
    count_y = 0

    #Add a minimum standard deviation for each point
    std_min = torch.tensor(1e-8)

    for data in dataset:
        mean_vec_x += torch.sum(data.x, dim=0)
        mean_vec_edge += torch.sum(data.edge_attr, dim=0)
        mean_vec_y += torch.sum(data.y, dim=0)
        std_vec_x += torch.sum(data.x**2, dim=0)
        std_vec_edge += torch.sum(data.edge_attr**2, dim=0)
        std_vec_y += torch.sum(data.y**2, dim=0)
        count_x += data.x.shape[0]
        count_edge += data.edge_attr.shape[0]
        count_y += data.y.shape[0]
        if (count_x > max_count or count_edge > max_count or count_y > max_count):
            break

    mean_vec_x /= count_x
    mean_vec_edge /= count_edge
    mean_vec_y /= count_y
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x / count_x - mean_vec_x**2), std_min)
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge / count_edge - mean_vec_edge**2), std_min)
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y / count_y - mean_vec_y**2), std_min)

    stat_list = [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y]
    return stat_list

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9
