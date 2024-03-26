from main import main_process


if __name__ == '__main__':
    task_zoo = [['CSA']]#, ['EAC'], ['EAC', 'ATL'], ['CSA', 'ATL']]# ]
    model_zoo = ['modelclass3d'] #'modelclass']#, 'convsharevit', 'vit', 'mobilevit', 'mobilenet']
    attn_zoo = ['normal'] # True, 
    site_zoo = [ ['Wrist', 'MCP'], ['Wrist'], ['MCP'], ['Wrist', 'MCP', 'Foot']]#,,]  #  
    for task in task_zoo:
        for model_counter in model_zoo:
            for site in site_zoo:
                batch_size = 20//len(site)
                if model_counter == 'convsharevit':
                    for attn in attn_zoo:
                        main_process(data_dir=r'/exports/lkeb-hpc/yanli/CSA_resplit/train',  target_category=task, 
                                    target_site=site, target_dirc=['TRA', 'COR'], phase='train',
                                    model_counter=model_counter, attn_type=attn, full_img=7, maxfold=5,
                                    test_dir=r'/exports/lkeb-hpc/yanli/ESMIRA_common/test')
                else:
                    main_process(data_dir=r'/exports/lkeb-hpc/yanli/CSA_resplit/train',  target_category=task, 
                                target_site=site, target_dirc=['TRA', 'COR'], phase='train',
                                model_counter=model_counter, attn_type='normal', full_img=7,  batch_size=10, maxfold=5,
                                test_dir=r'/exports/lkeb-hpc/yanli/ESMIRA_common/test')