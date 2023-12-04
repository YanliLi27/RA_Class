import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from generators.dataset_class import ESMIRA_generator
from train_func import train, pretrained, predict
from models.model import ModelClass
from myutils.output_finder import output_finder
from torchvision.models import mobilenet_v3_large
from models.mobilevit import mobilevit_s, mobilevit_xs, mobilevit_xxs
from myutils.record import record_save, corr_save, auc_save
import os


def main_process(data_dir='', target_category=['EAC', 'ATL'], 
                 target_site=['Wrist'], target_dirc=['TRA', 'COR'], phase='train'):
    best_auc_list = []
    dataset_generator = ESMIRA_generator(data_dir, target_category, target_site, target_dirc)
    for fold_order in range(5):
        save_task = target_category[0] if len(target_category)==1 else (target_category[0]+'_'+target_category[1])
        save_site = target_site[0] if len(target_site)==1 else (target_category[0]+'_'+target_category[1])
        save_father_dir = os.path.join('./models/figs', f'{save_site}_{save_task}')
        if not os.path.exists(save_father_dir):
            os.makedirs(save_father_dir)
        save_dir = os.path.join(save_father_dir, f'fold_{fold_order}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train_dataset, val_dataset = dataset_generator.returner(phase=phase, fold_order=fold_order, mean_std=False)
        # input: [N*5, 512, 512] + int(label)

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        in_channel = len(target_site) * len(target_dirc) * 5
        # model = ModelClass(in_channel, num_classes=2)
        # model = mobilenet_v3_large(last_channel=4096, inverted_residual_setting=, num_classes=2)
        model = mobilevit_xxs(img_2dsize=(512, 512), inch=in_channel, num_classes=2, patch_size=(4,4))

        # Step. 3 Train the model /OR load the weights
        output_name = output_finder(target_category, target_site, target_dirc, fold_order)
        if train_dataset is not None:
            best_auc = train(model=model, dataset=train_dataset, val_dataset=val_dataset, 
                             lr=0.001, num_epoch=50, batch_size=6, output_name=output_name,
                             extra_aug_flag=False, weight_decay=1e-5, optim_ada=False, save_dir=save_dir)
        corr_save(best_auc, 0, mode='acc', save_path=f'{save_dir}/record.txt')
        best_auc_list.append(best_auc)
        # Step. 4 Load the weights and predict
        model = pretrained(model=model, output_name=output_name)
        val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)
        G,P = predict(model, val_dataloader)
        print(classification_report(G,P))
        print(roc_auc_score(G,P))
    print(best_auc_list)


if __name__ == '__main__':
    task_zoo = [['CSA'], ['EAC'], ['EAC', 'ATL'], ['CSA', 'ATL']]
    for task in task_zoo:
        main_process(data_dir='D:\\ESMIRA\\ESMIRA_common',  target_category=task, 
                    target_site=['Wrist','MCP'], target_dirc=['TRA', 'COR'], phase='train')