import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from generators.dataset_class import ESMIRA_generator
from train_func import train, pretrained, predict, predictplus
from models.model import ModelClass, Classifier11
from models.model3d import ModelClass3D
from models.csv3d import make_csv3dmodel
from myutils.output_finder import output_finder
from torchvision.models import MobileNetV2
from models.vit import ViT
from models.mobilevit import mobilevit_s, mobilevit_xs, mobilevit_xxs
from models.convsharevit import make_csvmodel
from myutils.record import record_save, corr_save, auc_save
from myutils.log import Record
import os
from typing import Union, Literal


def main_process(data_dir='', target_category=['EAC', 'ATL'], 
                 target_site=['Wrist'], target_dirc=['TRA', 'COR'], phase='train',
                 model_counter='mobilevit', attn_type:Literal['normal', 'mobile', 'parr_normal', 'parr_mobile']='normal',
                 full_img:Union[bool, int]=5, batch_size:Union[int, None]=None,
                 maxfold:int=5, test_dir='D:\\ESMIRA\\CSA_resplit\\test', ls:float=0.1):
    best_auc_list = []
    best_test_list = []
    dataset_generator = ESMIRA_generator(data_dir, target_category, target_site, target_dirc, maxfold=maxfold)
    for fold_order in range(0, maxfold):
        save_task = target_category[0] if len(target_category)==1 else (target_category[0]+'_'+target_category[1])
        save_site = target_site[0] if len(target_site)==1 else (target_site[0]+'_'+target_site[1])
        save_father_dir = os.path.join('./models/figs', f'{model_counter}_{save_site}_{save_task}')
        if not os.path.exists(save_father_dir):
            os.makedirs(save_father_dir)
        save_dir = os.path.join(save_father_dir, f'fold_{fold_order}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dimenson = '3D' if '3d' in model_counter else '2D'
        train_dataset, val_dataset = dataset_generator.returner(phase=phase, fold_order=fold_order, 
                                                                mean_std=False, full_img=full_img, 
                                                                test_balance=True, dimension=dimenson)
        # input: [N*5, 512, 512] + int(label)

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        in_channel = len(target_site) * len(target_dirc) * full_img if isinstance(full_img, int) else len(target_site) * len(target_dirc) * 20
        # model = ModelClass(in_channel, num_classes=2)
        if model_counter == 'mobilenet':
            model = MobileNetV2(num_classes=2, inch=in_channel)
            batch_size = 6 if batch_size==None else batch_size
            lr = 0.0001
        elif model_counter == 'mobilevit':
            model = mobilevit_s(img_2dsize=(512, 512), inch=in_channel, num_classes=2, patch_size=(4,4))
            batch_size = 6 if batch_size==None else batch_size
            lr = 0.0001
        elif model_counter == 'vit':
            model = ViT(image_size=(512, 512), patch_size=(16, 16), num_classes=2, 
                        dim=256, depth=12, heads=8, mlp_dim=512, pool='mean', channels=in_channel, 
                        dropout=0.2, emb_dropout=0.2)
            batch_size = 6 if batch_size==None else batch_size
            lr = 0.0001
        elif model_counter == 'modelclass':
            model = ModelClass(in_channel, group_num=len(target_site) * len(target_dirc), num_classes=2)
            batch_size = 6 if batch_size==None else batch_size
            lr = 0.00005
        elif model_counter == 'modelclass11':
            model = ModelClass(in_channel, group_num=len(target_site) * len(target_dirc), num_classes=2, classifier=Classifier11)
            batch_size = 6 if batch_size==None else batch_size
            lr = 0.00005
        elif model_counter == 'convsharevit':
            model = make_csvmodel(img_2dsize=(512, 512), inch=in_channel, num_classes=2, num_features=43, extension=57, 
                  groups=(len(target_site) * len(target_dirc)), width=1, dsconv=False, attn_type=attn_type, patch_size=(2,2), 
                  mode_feature=False, dropout=True, init=False)
            batch_size = 12 if batch_size==None else batch_size
            lr = 0.00005
        elif model_counter == 'modelclass3d':
            in_ch=len(target_site)*len(target_dirc)
            if in_ch > 2:
                poolsize = 1
            else:
                poolsize = 3
            model = ModelClass3D(in_ch=in_ch, depth=in_channel//in_ch, group_num=len(target_site) * len(target_dirc), 
                                 num_classes=2, poolsize=poolsize)
            batch_size = 4 if batch_size==None else batch_size
            lr = 0.00005
        elif model_counter == 'csv3d':
            in_ch=len(target_site)*len(target_dirc)
            model = make_csv3dmodel(img_2dsize=(in_channel//in_ch, 512, 512), inch=in_ch, num_classes=2, num_features=43, extension=57, 
                  groups=(len(target_site) * len(target_dirc)), width=1, dsconv=False, attn_type=attn_type, patch_size=(2,2), 
                  mode_feature=False, dropout=True, init=False)
            batch_size = 4 if batch_size==None else batch_size
            lr = 0.00005
        else:
            raise ValueError('not supported model')
        
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)
        # Step. 3 Train the model /OR load the weights
        output_name = output_finder(model_counter, target_category, target_site, target_dirc, fold_order)
        if train_dataset is not None:
            best_auc = train(model=model, dataset=train_dataset, val_dataset=val_dataset, 
                             lr=lr, num_epoch=30, batch_size=batch_size, output_name=output_name,
                             extra_aug_flag=False, weight_decay=1e-2, optim_ada=True, save_dir=save_dir,
                             criterion=criterion)
            corr_save(best_auc, 0, mode='acc', save_path=f'{save_dir}/record.txt')
            best_auc_list.append(best_auc)
        # Step. 4 Load the weights and predict 
            # best auc
        model = pretrained(model=model, output_name=output_name)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=4)
        G,P, _ = predict(model, val_dataloader, criterion)
        print(classification_report(G,P))
        print(roc_auc_score(G,P))
        corr_save(confusion_matrix(G,P), 0, mode='cm', save_path=f'{save_dir}/record.txt')

        logger = Record('gt', 'pred', 'path', 'cm', 'auc')
        test_generator = ESMIRA_generator(test_dir, target_category, target_site, target_dirc, maxfold=maxfold)
        _, test_dataset = test_generator.returner(phase='test', fold_order=fold_order, mean_std=False, full_img=full_img, path_flag=True,
                                                  test_balance=False, dimension=dimenson)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        TG, TP, _, abs_path = predictplus(model, test_dataloader, criterion)
        # TG [batch, label], TP [batch, label], abs_path [batch, len(input), pathname]
        cm = confusion_matrix(TG, TP)
        auc = roc_auc_score(TG,TP)
        print('test classification report:', classification_report(TG,TP))
        print('test auc:', auc)
        best_test_list.append(auc)
        for i in range(len(TG)):
            logger(gt=TG[i], pred=TP[i], path=abs_path[i], cm=cm, auc=auc)
        logger.summary(save_path=f'{save_dir}/test_record.csv')
            # best loss
        # model2 = pretrained(model=model, output_name=output_name.replace('.model', 'loss.model'))
        # logger2 = Record('gt', 'pred', 'path', 'cm', 'auc')
        # TG2, TP2, _, abs_path2 = predictplus(model2, test_dataloader)
        # cm2 = confusion_matrix(TG2, TP2)
        # auc2 = roc_auc_score(TG2,TP2)
        # for i in range(len(TG2)):
        #     logger2(gt=TG2[i], pred=TP2[i], path=abs_path2[i], cm=cm2, auc=auc2)
        # logger2.summary(save_path=f'{save_dir}/test_record_loss.csv')

    print(best_auc_list)
    print('test auc:', best_test_list)


if __name__ == '__main__':
    dir = 'D:\\ESMIRA\\CSA_resplit\\aimira'# 'D:\\ESMIRA\\CSA_resplit\\train'
    task_zoo = [['CSA']]#, ['EAC'], ['EAC', 'ATL'], ['CSA', 'ATL'],]# ]
    model_zoo = ['modelclass']#'csv3d']#'modelclass3d']#'csv3d']#,  #'modelclass']#, 'convsharevit', 'vit', 'mobilevit', 'mobilenet']
    attn_zoo = ['normal'] # True, 
    site_zoo = [['Wrist']]#, ['Wrist', 'MCP']] #,,]  #  
    for task in task_zoo:
        for model_counter in model_zoo:
            for site in site_zoo:
                if model_counter == 'convsharevit':
                    for attn in attn_zoo:
                        main_process(data_dir=dir,  target_category=task, 
                                    target_site=site, target_dirc=['TRA', 'COR'], phase='train',
                                    model_counter=model_counter, attn_type=attn, full_img=7, maxfold=5)
                else:
                    main_process(data_dir=dir,  target_category=task, 
                                target_site=site, target_dirc=['TRA', 'COR'], phase='train',
                                model_counter=model_counter, attn_type='normal', full_img=7, maxfold=5)