import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from generators.dataset_class import ESMIRA_generator
from train_func import train, pretrained, predict
from models.model import ModelClass
from myutils.output_finder import output_finder
from torchvision.models import MobileNetV2
from models.vit import ViT
from models.mobilevit import mobilevit_s, mobilevit_xs, mobilevit_xxs
from models.convsharevit import make_csvmodel
from myutils.record import record_save, corr_save, auc_save
import os
from typing import Union, Literal


def main_process(data_dir='', target_category=['EAC', 'ATL'], 
                 target_site=['Wrist'], target_dirc=['TRA', 'COR'], phase='train',
                 model_counter='mobilevit', attn_type:Literal['normal', 'mobile', 'parr_normal', 'parr_mobile']='normal',
                 full_img:Union[bool, int]=5,
                 maxfold:int=5, test_dir='D:\\ESMIRA\\CSA_resplit\\test'):
    best_auc_list = []
    best_test_list = []
    dataset_generator = ESMIRA_generator(data_dir, target_category, target_site, target_dirc, maxfold=maxfold)
    for fold_order in range(maxfold):
        save_task = target_category[0] if len(target_category)==1 else (target_category[0]+'_'+target_category[1])
        save_site = target_site[0] if len(target_site)==1 else (target_site[0]+'_'+target_site[1])
        save_father_dir = os.path.join('./models/figs', f'{model_counter}_{save_site}_{save_task}')
        if not os.path.exists(save_father_dir):
            os.makedirs(save_father_dir)
        save_dir = os.path.join(save_father_dir, f'fold_{fold_order}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train_dataset, val_dataset = dataset_generator.returner(phase=phase, fold_order=fold_order, mean_std=False, full_img=full_img)
        # input: [N*5, 512, 512] + int(label)

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        in_channel = len(target_site) * len(target_dirc) * full_img if isinstance(full_img, int) else len(target_site) * len(target_dirc) * 20
        # model = ModelClass(in_channel, num_classes=2)
        if model_counter == 'mobilenet':
            model = MobileNetV2(num_classes=2, inch=in_channel)
            batch_size = 6
            lr = 0.0001
        elif model_counter == 'mobilevit':
            model = mobilevit_s(img_2dsize=(512, 512), inch=in_channel, num_classes=2, patch_size=(4,4))
            batch_size = 6
            lr = 0.0001
        elif model_counter == 'vit':
            model = ViT(image_size=(512, 512), patch_size=(16, 16), num_classes=2, 
                        dim=256, depth=12, heads=8, mlp_dim=512, pool='mean', channels=in_channel, 
                        dropout=0.2, emb_dropout=0.2)
            batch_size = 6
            lr = 0.0001
        elif model_counter == 'modelclass':
            model = ModelClass(in_channel, num_classes=2)
            batch_size = 6
            lr = 0.00005
        elif model_counter == 'convsharevit':
            model = make_csvmodel(img_2dsize=(512, 512), inch=in_channel, num_classes=2, num_features=43, extension=57, 
                  groups=(len(target_site) * len(target_dirc)), width=1, dsconv=False, attn_type=attn_type, patch_size=(2,2), mode_feature=False, dropout=False, init=False)
            batch_size = 8
            lr = 0.00005
        else:
            raise ValueError('not supported model')

        # Step. 3 Train the model /OR load the weights
        output_name = output_finder(model_counter, target_category, target_site, target_dirc, fold_order)
        if train_dataset is not None:
            best_auc = train(model=model, dataset=train_dataset, val_dataset=val_dataset, 
                             lr=lr, num_epoch=40, batch_size=batch_size, output_name=output_name,
                             extra_aug_flag=False, weight_decay=1e-2, optim_ada=True, save_dir=save_dir)
            corr_save(best_auc, 0, mode='acc', save_path=f'{save_dir}/record.txt')
            best_auc_list.append(best_auc)
        # Step. 4 Load the weights and predict
        model = pretrained(model=model, output_name=output_name)
        val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)
        G,P, _ = predict(model, val_dataloader)
        print(classification_report(G,P))
        print(roc_auc_score(G,P))
        corr_save(confusion_matrix(G,P), 0, mode='cm', save_path=f'{save_dir}/record.txt')

        # test_generator = ESMIRA_generator(test_dir, target_category, target_site, target_dirc, maxfold=maxfold)
        # _, test_dataset = test_generator.returner(phase='test', fold_order=fold_order, mean_std=False, full_img=full_img)
        # test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4)
        # TG, TP, _ = predict(model, test_dataloader)
        # print('test classification report:', classification_report(TG,TP))
        # print('test auc:', roc_auc_score(TG,TP))
        # best_test_list.append(roc_auc_score(TG,TP))
        # corr_save(roc_auc_score(TG,TP), 0, mode='acc', save_path=f'{save_dir}/test_record.txt')

    print(best_auc_list)
    print('test auc:', best_test_list)


if __name__ == '__main__':
    task_zoo = [['CSA']]#, ['EAC'], ['EAC', 'ATL'], ['C SA', 'ATL'],]# ]
    model_zoo = ['convsharevit']#, 'vit', 'mobilevit', 'mobilenet']
    attn_zoo = ['normal'] # True, 
    for task in task_zoo:
        for model_counter in model_zoo:
            if model_counter == 'convsharevit':
                for attn in attn_zoo:
                    main_process(data_dir='D:\\ESMIRA\\CSA_resplit\\train',  target_category=task, 
                                target_site=['Wrist','MCP'], target_dirc=['TRA', 'COR'], phase='train',
                                model_counter=model_counter, attn_type=attn, full_img=7, maxfold=5)
            else:
                main_process(data_dir='D:\\ESMIRA\\CSA_resplit\\train',  target_category=task, 
                            target_site=['Wrist','MCP'], target_dirc=['TRA', 'COR'], phase='train',
                            model_counter=model_counter, attn_type=attn, full_img=7, maxfold=5)