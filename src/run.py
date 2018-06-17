
import json
import argparse
import logging
import os

from generate_configure import generate_configure_file
from utils import train,test

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s -- %(message)s")
logger=logging.getLogger(__name__)

if __name__=='__main__':
    root_dir='../experiment_data'
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',type=str)
    parser.add_argument('--exp_dir',type=str)
    parser.add_argument('--pretrained',type=bool,default=False)
    parser.add_argument('--gen_config',type=bool,default=False)

    args=parser.parse_args()
    train_config_path =os.path.join(root_dir,args.exp_dir,'config_dir/train_configure.json')
    test_config_path = os.path.join(root_dir,args.exp_dir,'config_dir/test_configure.json')

    if args.gen_config:
        generate_configure_file(os.path.join(root_dir,args.exp_dir))

    if args.mode=='train':
        with open(train_config_path,'r') as f:
            opt=json.load(f)
            opt['pretrained'] = args.pretrained
            logger.info(opt)
            train(opt)

    elif args.mode=='test':
        with open(test_config_path,'r') as f:
            opt=json.load(f)
            test(opt)

    else:
        logger.info(args.mode+" illegal mode")

