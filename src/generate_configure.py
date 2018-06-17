import json
import os
import shutil

def generate_configure_file(exp_dir):
    data_dir=os.path.join('../experiment_data','data')
    exp_log_dir=os.path.join(exp_dir,'log_dir')
    exp_config_dir=os.path.join(exp_dir,'config_dir')
    exp_data_dir=os.path.join(exp_dir,'data_dir')

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        os.mkdir(exp_log_dir)
        os.mkdir(exp_config_dir)
        os.mkdir(exp_data_dir)

    if os.path.exists(exp_config_dir):
        shutil.rmtree(exp_config_dir)
    os.mkdir(exp_config_dir)


    train_opt = {
        'word_dime': 128,
        'hidd_dime': 256,
        'hidd_layer': 1,
        'epoches': 50,
        'valid_step':1000,
        'drop_rate': 0.2,
        'batch_size': 32,
        'RNN_type': 'GRU',
        'bidirectional': True,
        'teac_rate': 0.5,
        'lr': 0.001,
        'keep_ratio': 1,
        'exp_dir':exp_dir,
        'data_dir': data_dir,
        'log_dir':exp_log_dir,
    }

    test_opt = {
        'word_dime': 128,
        'hidd_dime': 256,
        'hidd_layer': 1,
        'drop_rate': 0,
        'batch_size': 1,
        'RNN_type': 'GRU',
        'bidirectional': True,
        'teac_rate': 0,
        'keep_ratio': 1,
        'exp_dir':exp_dir,
        'data_dir': data_dir,
        'log_dir':exp_log_dir,
        'exp_data':exp_data_dir
    }
    train_confi_file_path=os.path.join(exp_config_dir,'train_configure.json')
    test_confi_file_path=os.path.join(exp_config_dir,'test_configure.json')
    with open(train_confi_file_path,'w') as f:
        json.dump(train_opt,f)

    with open(test_confi_file_path,'w') as f:
        json.dump(test_opt,f)

