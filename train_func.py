import os  # for paths
import torch.nn as nn   # used for bulid the neural networks
from tqdm import tqdm  # just for visualization
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
from myutils.extra_aug import extra_aug
from thop import profile
from myutils.record import record_save, corr_save, auc_save


def train_step(model, optimizer, criterion, train_loader, extra_aug_flag:bool=False, epoch:int=0,
               device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    avg_loss = []
    if extra_aug_flag:
        crit_aug = nn.MSELoss()
        for x,y in tqdm(train_loader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                x = x.to(device)
                aug_x = extra_aug(x)
                y = y.to(device)
                y_pred = model(x=x)
                y_aug_pred = model(aug_x)
                loss_self = crit_aug(y_aug_pred, y_pred) * (0.01*epoch)
                loss = criterion(y_pred, y) + loss_self
                # next three lines are unchanged for all the tasks
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            avg_loss.append(loss.item())
    
    else:
        for x,y in tqdm(train_loader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                # next three lines are unchanged for all the tasks
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            avg_loss.append(loss.item())
    return sum(avg_loss)/len(avg_loss)


def predict(model, test_loader, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for x,y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            y_pred = torch.argmax(pred, dim=1)
            total_preds = torch.cat((total_preds, y_pred.cpu()), 0)
            total_labels = torch.cat((total_labels, y.cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def train(model, dataset, val_dataset, lr=0.0001, num_epoch:int=100, batch_size:int=10, 
          output_name:str='', extra_aug_flag:bool=False, weight_decay:float=1e-5,
          optim_ada:bool=False, save_dir:str='',
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    lr = lr
    # weight_dec = 0.0001
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_dec)
    if optim_ada:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    criterion = nn.CrossEntropyLoss()
    batch_size = batch_size

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    max_metric = 0

    x1, _ = next(iter(dataloader))
    print(x1.shape)
    flops, params = profile(model, inputs=(x1, ))
    print(f'FLOPS: {flops}, params: {params}')

    model = model.to(device)

    model_file_name = output_name
    recorded_flag = False
    
    for epoch in range(1, num_epoch + 1):
        train_loss = train_step(model, optimizer, criterion, dataloader, extra_aug_flag, epoch)
        print(train_loss)
        if epoch % 50 == 0:
            print(f"Loss at epoch {epoch} is {train_loss}")
        G,P = predict(model, val_dataloader)
        # accuracy = accuracy_score(G, P)
        auc = roc_auc_score(G, P)
        # fpr, tpr, thresholds = roc_curve(G, P)
        f1_scores = f1_score(G, P)
            
        if auc > max_metric and train_loss<=0.67:
            max_metric = auc
            recorded_flag = True
            torch.save(model.state_dict(), model_file_name)
            print("saving best model with auc: ", auc)
            auc_save(max_metric, epoch, save_path=f'{save_dir}/record.txt')
            auc_save(f1_scores, epoch, save_path=f'{save_dir}/record.txt', mode='f1')
        else:
            print('auc:',auc)
            if not optim_ada:
                scheduler.step()
        print('f1:', f1_scores)

        if epoch >20 and recorded_flag==False:
            recorded_flag = True
            print('save for compare')
            max_metric = auc
            auc_save(max_metric, epoch, save_path=f'{save_dir}/record.txt')
            auc_save(f1_scores, epoch, save_path=f'{save_dir}/record.txt', mode='f1')
    return max_metric
            
            


def pretrained(model, output_name:str=''):
    # load the weight
    model_file_name = output_name
    if os.path.isfile(model_file_name):
        checkpoint = torch.load(model_file_name)
        model.load_state_dict(checkpoint)
    return model


    