import argparse
import os
import os.path as osp
from torch import nn
import sys
import pickle
root = os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
sys.path.append(root)
from main import initialization
# print(root)
# print(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))))
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/models/")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
from modeling import models
import torch
import numpy as np
import torch.utils.data as tordata

from data.transform import get_transform
from data.collate_fn import CollateFn
from data.dataset import DataSet
import data.sampler as Samplers
from utils import get_valid_args,  np2var, list2var, get_attr_from

def make_parser():
    parser = argparse.ArgumentParser("Gait Part Demo!")
##############################################
    parser.add_argument(
        "--gait_model",default="/home/jdy/Gaitdateset/gait_model/Baseline-150000.pt",help="path of gait model"
    )
    
    # 有剪影图的pkl
    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--pkl_save_path", default="/home/jdy/Gaitdateset/Image03/videos/", help="path to save pkl"
    )

    # 有特征矩阵的pkl
    parser.add_argument(
        # "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--whole_pkl_save_path", default="/home/jdy/Gaitdateset/Image04/videos/", help="path to save pkl"
    )
    parser.add_argument('-nw', '--n_workers', default=4, type=int, help='Number of thread workers. Default: 4')
    parser.add_argument('--cfgs', type=str,
                    default='/home/jdy/OpenGait/configs/baseline/baseline_OUMVLP.yaml', help="path of config file")
#############################################
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    parser.add_argument("--img_size",type=int, default=64, help="img_size")
    parser.add_argument("--workers",type=int, default=4, help="workers")
    parser.add_argument("--verbose",type=bool, default=False, help="verbose")
    parser.add_argument("--dataset",default='CASIAB', help="dataset")



    parser.add_argument('--local_rank', type=int, default=0,
                        help="passed by torch.distributed.launch module")
    parser.add_argument('--phase', default='train',
                        choices=['train', 'test'], help="choose train or test phase")
    parser.add_argument('--log_to_file', action='store_true',
                        help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
    parser.add_argument('--iter', default=0, help="iter to restore")

    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )


    return parser


### get model

torch.distributed.init_process_group('nccl', init_method='env://')


def loadModel(model_type, cfg_path):
    Model = getattr(models, model_type)
    cfgs = config_loader(cfg_path)
    model = Model(cfgs, training=False)
    #model._load_ckpt(savepath)
    return model, cfgs


cfgs = {  "gaitmodel":{
    "model_type": "Baseline",
    "cfg_path": "OpenGait/configs/baseline/baseline.yaml",
},
}


class Gait(object):
    def __init__(
        self,
        model,
        device=torch.device("cpu")
    ):
        self.model = model
        self.device = device
        self.id = 0
        
    def inference(self, inputs, path):
        # inputs = inputs_pretreament(inputs)

        # real_inputs = torch.tensor(torch.tensor(input), None, None, None, None)
        
        outputs = self.model(inputs)
        embs = outputs["inference_feat"]
        #########print
        print(embs)
        outputs = [inputs, embs]
        # path = "/home/jdy/Gaitdateset/Image04/videos/"
        save_name = "{}{}.pkl".format(path, self.id)
        pkl = open(save_name, 'wb')
        pickle.dump(outputs, pkl)
        self.id += 1
        return embs



def get_loader(cfgs):
    sampler_cfg = cfgs['evaluator_cfg']['sampler']
    dataset = DataSet(cfgs['data_cfg'], False)

    Sampler = get_attr_from([Samplers], sampler_cfg['type'])
    vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
        'sample_type', 'type'])
    sampler = Sampler(dataset, **vaild_args)

    loader = tordata.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=CollateFn(dataset.label_set, sampler_cfg),
        num_workers=cfgs['data_cfg']['num_workers'])
    return loader

def inputs_pretreament(cfgs,inputs):
    seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
    trf_cfgs = cfgs['evaluator_cfg']['transform']
    seq_trfs = get_transform(trf_cfgs)
    if len(seqs_batch) != len(seq_trfs):
        raise ValueError(
            "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
    requires_grad = False
    seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
            for trf, seq in zip(seq_trfs, seqs_batch)]

    typs = typs_batch
    vies = vies_batch

    labs = list2var(labs_batch).long()

    if seqL_batch is not None:
        seqL_batch = np2var(seqL_batch).int()
    seqL = seqL_batch

    if seqL is not None:
        seqL_sum = int(seqL.sum().data.cpu().numpy())
        ipts = [_[:, :seqL_sum] for _ in seqs]
    else:
        ipts = seqs
    del seqs
    return ipts, labs, typs, vies, seqL


def gait(args):
    pkl_save = args.pkl_save_path
    # cfgs = config_loader(args.cfgs)
    # torch.distributed.init_process_group('nccl', init_method='env://')
    if 1:
        print("========= Loading model..... ==========")
        initialization(config_loader(cfgs["gaitmodel"]["cfg_path"]), False)
        gaitmodel, newcfgs = loadModel(**cfgs["gaitmodel"])
        gaitmodel.requires_grad_(False)
        gaitmodel.eval()
        print("========= Load Done.... ==========")

        # gait_model_path = args.gait_model
        # # gait_model = torch.load(gait_model_path, map_location="cpu")
        # # initialization
        # engine_cfg = cfgs['evaluator_cfg']
        # output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
        #                         cfgs['model_cfg']['model'], engine_cfg['save_name'])
        # print("################")
        # print(output_path)

        # msg_mgr = get_msg_mgr()
        # msg_mgr.init_logger(output_path, args.log_to_file)
        # msg_mgr.log_info(engine_cfg)
        # # print(args.cfgs)
        # seed = torch.distributed.get_rank()
        # init_seeds(seed)


        # # print(1)
        # # cfgs = config_loader(args.cfgs)
        # print(3)
        # model_cfg = cfgs['model_cfg']
        # print(3)
        # Model = getattr(models, model_cfg['model'])
        # print(2)
        # gait_model = Model(cfgs, False)
        # if cfgs['trainer_cfg']['fix_BN']:
        #     gait_model.fix_BN()
        # gait_model = get_ddp_module(gait_model)
        

        # # gait_model = Baseline.build_network(args.cfgs['model_cfg'])
        # weight = torch.load(gait_model_path, map_location="cpu")
        # gait_model.load_state_dict(weight)
    # except OSError:
    #     print("Can't load GaitModel")

    gait = Gait(gaitmodel, args.device)

    # for id in sorted(os.listdir(pkl_save)):
    #     #type-seq应该是一个目录!
    #     for type in sorted(os.listdir(os.path.join(pkl_save,id))):
    #         for seq in sorted(os.listdir(os.path.join(pkl_save,id,type))):
    #             for filename01 in os.listdir(os.path.join(pkl_save,id,type)):
    #                 for filename in os.listdir(os.path.join(pkl_save,id,type,filename01)):
    #                     print(111111111111)
    #                     print(filename)
    #                     print(111111111111)
    #                     if filename.endswith('.pkl'):
    #                         with open(os.path.join(pkl_save,id,type,filename01,filename), 'rb') as f:
    #                             input = pickle.load(f)
    #                         f.close()
    #                     else:
    #                         print("Can't find the pkl file!")
    #                         continue
    #                     save_path = os.path.join(args.whole_pkl_save_path, id, type, seq,'.pkl')
    #                     embs = gait.inference(input,save_path)
    #                     del input
    loader = get_loader(newcfgs)
    print(len(loader))

    for inputs in loader:
        print(123456787)
        ipts = inputs_pretreament(newcfgs, inputs)
        # with autocast(enabled=self.engine_cfg['enable_float16']):
        #     embs = gait.inference(ipts, save_path)
        #     del ipts
        embs = gait.inference(ipts, args.whole_pkl_save_path)
        print("#############")
        print(embs)
        del ipts


if __name__ == "__main__":
    args = make_parser().parse_args()
    gait(args)