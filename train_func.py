import os  # for paths
import platform
import torch.nn as nn   # used for bulid the neural networks
from tqdm import tqdm  # just for visualization
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, classification_report, confusion_matrix
from myutils.extra_aug import extra_aug
from thop import profile
from myutils.record import record_save, corr_save, auc_save
from myutils.log import Record


def train_step(model, optimizer, criterion, train_loader, extra_aug_flag:bool=False, epoch:int=0,
               device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    avg_loss = []
    if extra_aug_flag:
        crit_aug = nn.MSELoss()
        train_loader = tqdm(train_loader) if platform.system().lower()=='windows' else train_loader
        for x,y in train_loader:
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
        train_loader = tqdm(train_loader) if platform.system().lower()=='windows' else train_loader
        for x,y in train_loader:
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


def predict(model, test_loader, criterion=None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    avg_loss = []
    with torch.no_grad():
        test_loader = tqdm(test_loader) if platform.system().lower()=='windows' else test_loader
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            avg_loss.append(loss.item())
            y_pred = torch.argmax(pred, dim=1)
            total_preds = torch.cat((total_preds, y_pred.cpu()), 0)
            total_labels = torch.cat((total_labels, y.cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten(), sum(avg_loss)/len(avg_loss)


def predictplus(model, test_loader, criterion=None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    abs_path = []
    avg_loss = []
    with torch.no_grad():
        test_loader = tqdm(test_loader) if platform.system().lower()=='windows' else test_loader
        for x,y,z in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            avg_loss.append(loss.item())
            y_pred = torch.argmax(pred, dim=1)
            total_preds = torch.cat((total_preds, y_pred.cpu()), 0)
            total_labels = torch.cat((total_labels, y.cpu()), 0)
            paths = [(z[i], pred[i]) for i in range(len(z))]
            abs_path.extend(paths)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten(), sum(avg_loss)/len(avg_loss), abs_path


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

    # record
    logger = Record('trainloss', 'valloss', 'f1', 'cm', 'metric')
    logger_best = Record('trainloss', 'valloss', 'f1', 'cm', 'metric')
    max_metric = 0
    min_metric = 20000.0
    
    # thop will cause the model load failure
    # x1, _ = next(iter(dataloader))
    # print(x1.shape)
    # flops, params = profile(model, inputs=(x1, ))
    # print(f'FLOPS: {flops}, params: {params}')

    model = model.to(device)

    model_file_name = output_name
    recorded_flag = False
    
    for epoch in range(1, num_epoch + 1):
        train_loss = train_step(model, optimizer, criterion, dataloader, extra_aug_flag, epoch)
        print(f'epoch {epoch}, train loss: {train_loss}')
        if epoch % 50 == 0:
            print(f"Loss at epoch {epoch} is {train_loss}")
        G,P, val_loss = predict(model, val_dataloader, criterion)
        # accuracy = accuracy_score(G, P)
        auc = roc_auc_score(G, P)
        # fpr, tpr, thresholds = roc_curve(G, P)
        f1_scores = f1_score(G, P)
        cm = confusion_matrix(G,P)
        logger(trainloss=train_loss, valloss=val_loss, f1=f1_scores, cm=cm, metric=auc)
        
        if auc > max_metric and train_loss<=0.69 and val_loss<1.0:
            max_metric = auc
            recorded_flag = True
            torch.save(model.state_dict(), model_file_name)
            print("saving best model with auc: ", auc)
            logger_best(epoch=epoch, trainloss=train_loss, valloss=val_loss, f1=f1_scores, cm=confusion_matrix(G,P), metric=max_metric)
        else:
            print('auc:',auc)
            if not optim_ada:
                scheduler.step()

        if val_loss < min_metric and train_loss<=0.69:
            min_metric = val_loss
            recorded_flag = True
            torch.save(model.state_dict(), model_file_name.replace('.model', 'loss.model'))
        print('f1:', f1_scores)
        print('val loss:', val_loss)

        if epoch >20 and recorded_flag==False:
            recorded_flag = True
            print('save for compare')
            max_metric = auc
            auc_save(train_loss, epoch, save_path=f'{save_dir}/record.txt', mode='train loss')
            auc_save(val_loss, epoch, save_path=f'{save_dir}/record.txt', mode='val loss')
            auc_save(f1_scores, epoch, save_path=f'{save_dir}/record.txt', mode='f1')
            corr_save(confusion_matrix(G,P), 0, mode='cm', save_path=f'{save_dir}/record.txt')
            auc_save(max_metric, epoch, save_path=f'{save_dir}/record.txt')
        
        if epoch%15 ==0:
            logger.summary(f'{save_dir}/record.csv')
            logger_best.summary(f'{save_dir}/record_best.csv')
    logger.summary(f'{save_dir}/record.csv')
    logger_best.summary(f'{save_dir}/record_best.csv')
    return max_metric
            

def pretrained(model, output_name:str=''):
    # load the weight
    model_file_name = output_name
    if os.path.isfile(model_file_name):
        checkpoint = torch.load(model_file_name)
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f'the weights dont exist {model_file_name}')
    return model


    