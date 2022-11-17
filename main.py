import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
from generators.dataset_class import ESMIRA_generator
from train_func import train, pretrained, predict
from models.model import ModelClass
from utils.output_finder import output_finder


def main_process(data_dir='', target_category=['EAC', 'ATL'], 
                 target_site=['Wrist'], target_dirc=['TRA', 'COR'], phase='train'):

    dataset_generator = ESMIRA_generator(data_dir, target_category, target_site, target_dirc)
    for fold_order in range(5):
        train_dataset, val_dataset = dataset_generator.returner(phase=phase, fold_order=fold_order, mean_std=False)
        # input: [N*5, 512, 512] + int(label)

        # Step. 2 get the model: (can be any nn.Module, make sure it fit your input size and output size)
        in_channel = len(target_site) * len(target_dirc) * 5
        model = ModelClass(in_channel, num_classes=2) 

        # Step. 3 Train the model /OR load the weights
        output_name = output_finder(target_category, target_site, target_dirc, fold_order)
        if train_dataset is not None:
            train(model=model, dataset=train_dataset, val_dataset=val_dataset, 
                  lr=0.0001, num_epoch=100, num_classes=2, output_name=output_name,
                  extra_aug=False)

        # Step. 4 Load the weights and predict
        model = pretrained(model=model, output_name=output_name)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        G,P = predict(model, val_dataloader)
        print(classification_report(G,P))
        print(roc_auc_score(G,P))


if __name__ == '__main__':
    main_process(data_dir='',  target_category=['EAC', 'CSA', 'ATL'], 
                 target_site=['Wrist', 'MCP', 'Foot'], target_dirc=['TRA', 'COR'], phase='train')