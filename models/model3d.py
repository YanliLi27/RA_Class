import torch
import torch.nn as nn


class conv_block_group(nn.Module):
    def __init__(self, ch_in, ch_out, group_num=2):
        super(conv_block_group, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=group_num),
            nn.BatchNorm3d(ch_out),
            nn.SiLU(True),
            nn.Conv3d(ch_out, ch_out, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, groups=group_num),
            nn.BatchNorm3d(ch_out),
            nn.SiLU(True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder3d(nn.Module):
    def __init__(self, in_ch:int=2, group_num:int=2):  # 6 is 3 TRA + 3 COR
        super(Encoder3d, self).__init__()
        self.Maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.Conv1 = conv_block_group(ch_in=in_ch, ch_out=16*group_num, group_num=group_num)
        self.Conv2 = conv_block_group(ch_in=16*group_num, ch_out=32*group_num, group_num=group_num)
        self.Conv3 = conv_block_group(ch_in=32*group_num, ch_out=64*group_num, group_num=group_num)
        self.Conv4 = conv_block_group(ch_in=64*group_num, ch_out=128*group_num, group_num=group_num)
        # [256*2, 7, 64, 64]

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        return x4


class Classifier(nn.Module):
    def __init__(self, num_classes=2, depth:int=7, node:int=14):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d((depth, 1, 1))
        self.classifier_fc = nn.Sequential(
            nn.Linear(128 * node, 4096),
            nn.SiLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.SiLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x) # [256*2, 7, 64, 64] --> [256*2, 7, 1, 1]
        x = torch.flatten(x, 1)  # [256*2, 7, 1, 1]  --> [256*27]
        x = self.classifier_fc(x)
        # x = self.softmax(x)
        return x


class ModelClass3D(nn.Module):
    def __init__(self, in_ch:int=2, depth:int=14, group_num:int=2, num_classes=2, encoder=Encoder3d, classifier=Classifier, init_weights: bool = True):
        super(ModelClass3D, self).__init__()
        self.encoder_class = encoder(in_ch=in_ch, group_num=group_num)
        self.classifier = classifier(num_classes=num_classes, depth=depth, node=in_ch*depth)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # encoding path
        x = self.encoder_class(x)
        # decoding + concat path
        d = self.classifier(x)
        return d

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
