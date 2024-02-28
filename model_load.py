from models.convsharevit import make_csvmodel
from train_func import pretrained
from myutils.output_finder import output_finder


model = make_csvmodel(img_2dsize=(512, 512), inch=14, num_classes=2, num_features=43, extension=57, 
                  groups=2, width=1, dsconv=False, attn_type='normal', patch_size=(2,2), 
                  mode_feature=False, dropout=True, init=False)
output_name = output_finder('convsharevit', ['CSA'], ['Wrist'], ['TRA', 'COR'], 0)
print(output_name)
model = pretrained(model=model, output_name=output_name)