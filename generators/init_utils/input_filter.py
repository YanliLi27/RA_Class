

def input_filter(common_cs_dict:dict, target_site:list, target_dirc:list)->dict:
    keyname = common_cs_dict.keys()  # ['EAC_XXX_XXX', 'CSA_XXX_XXX']
    input_limit = []
    for site in target_site:
        for dirc in target_dirc:
            input_limit.append(f'{site}_{dirc}')
    
    for name in keyname:
        split_name = name.split('_')
        site_dirc_name = f'{split_name[1]}_{split_name[2]}'
        if site_dirc_name in input_limit:
            continue
        else:
            del common_cs_dict[name]
    return common_cs_dict