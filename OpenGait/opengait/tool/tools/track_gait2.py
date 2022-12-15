import argparse
import os
import os.path as osp
import pickle
import time
import cv2
import torch
import glob
import pandas as pd
from torch import nn
import torch

from loguru import logger
import sys
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tracking_utils.timer import Timer
from pathlib import Path
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
import shutil

root = os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
sys.path.append(root)
config = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))))) + "/PaddleSeg/contrib/PP-HumanSeg/"
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))))) + "/PaddleSeg/contrib/PP-HumanSeg/src")
from seg_demo import seg_opengait_image
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))) + "/datasets")
from pretreatment import pretreat
from utils import config_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/models/")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
from modeling import models
import gait_compare as gc
from changejson import writejson, getid, getidlist


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# 运行指令：
# python3 OpenGait/opengait/tool/tools/track_gait2.py video -f ByteTrack/exps/example/mot/yolox_x_mix_det.py -c ByteTrack/pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result  --phase test --compare True

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "--jsonpath", default="./OpenGait/datasets/CASIA-B/demo.json", help="path to json"
    )
    # 初始视频gallery作用，具体到.mp4文件,有机会改成--gallerypath
    parser.add_argument(
        "--gallerypath", default="./Image/palace.mp4", help="path to images or video"
    )
    # 有剪影图 png的
    parser.add_argument(
        "--gallerysave_path", default="./Gallery/Image01/videos/", help="path to save"
    )
    # 只有剪影图
    parser.add_argument(
        "--gallerysavesil_path", default="./Gallery/Image02/videos/", help="path to save"
    )
    
    # 有剪影图的pkl
    parser.add_argument(
        "--gallerypkl_save_path", default="./Gallery/Image03/videos/", help="path to save pkl"
    )

    # 初始视频probe作用，具体到.mp4文件,有机会改成--probepath
    parser.add_argument(
        "--probepath", default="./Image/005-nm-01-090.mp4", help="path to images or video"
    )
    # 有剪影图 png的
    parser.add_argument(
        "--probesave_path", default="./Probe/Image01/videos/", help="path to save"
    )
    # 只有剪影图
    parser.add_argument(
        "--probesavesil_path", default="./Probe/Image02/videos/", help="path to save"
    )

        # 有剪影图的pkl
    parser.add_argument(
        "--probepkl_save_path", default="./Probe/Image03/videos/", help="path to save pkl"
    )
##############################################
    parser.add_argument(
        "--gait_model",default="./gait_model/Baseline-150000.pt",help="path of gait model"
    )
    

    # 有特征矩阵的pkl
    parser.add_argument(
        "--whole_pkl_save_path", default="./Image04/videos/", help="path to save pkl"
    )
    # 生成embs.pkl的位置
    parser.add_argument(
        "--embspath", default="./Embs/", help="path to save or load embs"
    )
    parser.add_argument('-nw', '--n_workers', default=4, type=int, help='Number of thread workers. Default: 4')
    parser.add_argument('--cfgs', type=str,
                    default='./OpenGait/configs/baseline/baseline.yaml', help="path of config file")
#############################################
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    
    parser.add_argument("--img_size",type=int, default=64, help="img_size")
    parser.add_argument("--workers",type=int, default=4, help="workers")
    parser.add_argument("--verbose",type=bool, default=False, help="verbose")
    parser.add_argument("--dataset",default='CASIAB', help="dataset")
    # gait
    parser.add_argument('--local_rank', type=int, default=0,
                        help="passed by torch.distributed.launch module")
    parser.add_argument('--phase', default='train',
                        choices=['train', 'test'], help="choose train or test phase")
    parser.add_argument('--log_to_file', action='store_true',
                        help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
    parser.add_argument('--iter', default=0, help="iter to restore")

    # compare
    parser.add_argument('--comparegait', type=bool, default=False, help="whether to compare")

    return parser


def loadModel(model_type, cfg_path, compare=False):
    Model = getattr(models, model_type)
    cfgs = config_loader(cfg_path)
    model = Model(cfgs, training=False)
    model.compare = compare
    #model._load_ckpt(savepath)
    return model, cfgs


cfgs = {  "gaitmodel":{
    "model_type": "Baseline",
    "cfg_path": "OpenGait/configs/baseline/baseline.yaml",
},
}

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info


def imageflow_demo(predictor, vis_folder, current_time, args, gallery=True):
    if gallery:
        video = args.gallerypath
    else:
        video = args.probepath
    # save_path = osp.join(args.save_path, id, type, people)
    # temp = "根据trackid生成的tid和写死nm-01和000的三级目录"
    # save_path = osp.join(args.save_path, temp) # 再加一个三级目录信息的路径，因为是要求多人视频了
    # savesil_path = osp.join(args.savesil_path, temp)
    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)

    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    ids = []
    # 如果一个人的id是1然后走出了视频里面，那下一个进来的人id是1还是2????
    while True:
        if frame_id == 12:
            break
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()

        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                        # 清除一下目录, 并且把第一次出现的tid加入ids里面
                        tidstr = "{:03d}".format(tid)
                        if gallery:
                            save_path = osp.join(args.gallerysave_path, tidstr, "nm-01", "000") # 再加一个三级目录信息的路径，因为是要求多人视频了
                            savesil_path = osp.join(args.gallerysavesil_path, tidstr, "nm-01", "000")
                        else:
                            save_path = osp.join(args.probesave_path, tidstr, "nm-01", "000") # 再加一个三级目录信息的路径，因为是要求多人视频了
                            savesil_path = osp.join(args.probesavesil_path, tidstr, "nm-01", "000")
                        if tid not in ids:
                            if os.path.exists(savesil_path):
                                shutil.rmtree(savesil_path)
                            if os.path.exists(save_path):
                                shutil.rmtree(save_path)
                            ids.append(tid)

                        
                        x = tlwh[0]
                        y = tlwh[1]
                        width = tlwh[2]
                        height = tlwh[3]

                        x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
                        w, h = x2 - x1, y2 - y1
                        x1_new = max(0, int(x1 - 0.1 * w))
                        x2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(x2 + 0.1 * w))
                        y1_new = max(0, int(y1 - 0.1 * h))
                        y2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(y2 + 0.1 * h))
                        new_w = x2_new - x1_new
                        new_h = y2_new - y1_new
                        tmp = frame[y1_new: y2_new, x1_new: x2_new, :]
                        
                        final_config = config + "inference_models/human_pp_humansegv1_lite_192x192_inference_model_with_softmax/deploy.yaml"
            
                        # 先用tid来进行命名
                        save_name = "{:03d}-{:03d}.png".format(tid, frame_id)
                        print(final_config, save_name)
                        # if gallery:
                        #     save_path = osp.join(args.gallerysave_path, str(tid), "nm-01", "000") # 再加一个三级目录信息的路径，因为是要求多人视频了
                        #     savesil_path = osp.join(args.gallerysavesil_path, str(tid), "nm-01", "000")
                        # else:
                        #     save_path = osp.join(args.probesave_path, str(tid), "nm-01", "000") # 再加一个三级目录信息的路径，因为是要求多人视频了
                        #     savesil_path = osp.join(args.probesavesil_path, str(tid), "nm-01", "000")
                        seg_opengait_image(tmp, final_config, save_name, savesil_path,save_path)

                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    
    # ids去重, 以后再加个升序排列
    ids=list(set(ids))

    # 写入要求的json文件里面
    writejson(args.jsonpath, ids, gallery)


def gait(args, embsdict:dict, judge=False):
    print("========= Loading model..... ==========")
    # initialization(config_loader(cfgs["gaitmodel"]["cfg_path"]), False)
    gaitmodel, newcfgs = loadModel(**cfgs["gaitmodel"])
    gaitmodel.requires_grad_(False)
    gaitmodel.eval()
    print("========= Load Done.... ==========")

    # loader = gaitmodel.test_loader
    cfgloader = config_loader(cfgs["gaitmodel"]["cfg_path"])
    print(cfgloader)
    loader = gaitmodel.get_loader(
                cfgloader['data_cfg'], train=False, compare=judge)
    # dic = []
    i = 0
    for inputs in loader:
        print("num: ", i)
        ipts = gaitmodel.inputs_pretreament(inputs)
        print("ipts#######################")
        print("ipts: ",ipts)
        print("inputs: ", inputs[1][0], inputs[2], inputs[3])
        id = inputs[1][0]+1
        type = inputs[2][0] 
        view = inputs[3][0]
        savePklPath = "{}/{:03d}/{}/{}".format(args.whole_pkl_save_path,id,type,view)
        if not os.path.exists(savePklPath):
            os.makedirs(savePklPath)
        savePklName = "{}/{}.pkl".format(savePklPath, inputs[3][0])
        retval, embs = gaitmodel.forward(ipts)
        pkl = open(savePklName, 'wb')
        pickle.dump(embs, pkl)
        # dic.append[embs]
        if id not in embsdict:
            embsdict[id] = []

        ap = {}
        ap[type] = {}
        ap[type][view] = embs
        # embsdict[id].append(ap) 
        
        if len(embsdict[id]) == 0:
            embsdict[id].append(ap)
        else:
            for idx, e in enumerate(embsdict[id]):
                # print("idx: ",idx)
                # print("id: ",id)
                # print("type: ",type)
                # print("e: ",e)
                # print("ap: ",ap)
                # if operator.eq(e, ap):
                if str(e) == str(ap):
                    print("############# equal so break ##############")
                    break
                elif idx == len(embsdict[id])-1:
                    embsdict[id].append(ap)            
        del ipts
        i+=1

def gaitcompare(args, embsdict:dict, judge=False):
    print("========= Begin comparing..... ==========")
    print("========= Loading model..... ==========")
    # initialization(config_loader(cfgs["gaitmodel"]["cfg_path"]), False)
    gaitmodel, newcfgs = loadModel(**cfgs["gaitmodel"])
    gaitmodel.requires_grad_(False)
    gaitmodel.eval()
    print("========= Load Done.... ==========")

    # loader = gaitmodel.test_loader
    cfgloader = config_loader(cfgs["gaitmodel"]["cfg_path"])
    print(cfgloader)
    loader = gaitmodel.get_loader(
                cfgloader['data_cfg'], train=False, compare=judge)
    # dic = []
    print(loader)

    for inputs in loader:
        ipts = gaitmodel.inputs_pretreament(inputs)
        print("inputs: ", inputs[1][0], inputs[2], inputs[3])
        # subject:inputs[1][0] type:inputs[2][0] view:inputs[3][0]
        print("################")
        # print("ipts: ",ipts)
        # print("################")
        retval, embs = gaitmodel.forward(ipts)
        # print("####################retval")
        # print(retval)
        id = gc.compareid(retval,embsdict,1000)
        print("#################################id")
        print(id)
        realid = getid(args.jsonpath, id)
        print(realid)
        del ipts

def deletedirs(args, gallery=True):
    idlist = getidlist(args.jsonpath, gallery)
    print("######### gallery #####################")
    print(gallery)
    print("######### idlist ##################")
    print(idlist)
    if gallery:
        path = args.gallerysave_path
        silpath = args.gallerysavesil_path
    else:
        path = args.probesave_path
        silpath = args.probesavesil_path
    print("########## path dir ##########################")
    for dir in os.listdir(path):
        if dir in idlist:
            print(dir, "in idlist")
        else:
            shutil.rmtree(osp.join(path, dir))
    print("########## silpath dir ##########################")
    for dir in os.listdir(silpath):
        if dir in idlist:
            print(dir, "in idlist")
        else:
            print(dir)
            shutil.rmtree(osp.join(silpath, dir))

    
    print("delete dir")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    # del model
    current_time = time.localtime()
    
    # 视频输入不能是三级目录了就是一个mp4的video, 目前是gallery部分
    if args.gallerypath.endswith(".mp4"):
        imageflow_demo(predictor, vis_folder, current_time, args)
    else:
        print("gallery video is not end with .mp4")
        sys.exit()
    
    # path要改，前面加个gallery的
    print("###################")
    print(Path(args.gallerypkl_save_path))
    print("###################")
    deletedirs(args, True)
    pretreat(input_path=Path(args.gallerysave_path), output_path=Path(args.gallerypkl_save_path), img_size=args.img_size, 
                                workers=args.n_workers, verbose=args.verbose, dataset=args.dataset)

    if args.probepath.endswith(".mp4"):
        imageflow_demo(predictor, vis_folder, current_time, args, False)
    else:
        print("probe video is not end with .mp4")
        sys.exit()
    print("###################")
    print(Path(args.probepkl_save_path))
    print("###################")
    deletedirs(args, False)
    pretreat(input_path=Path(args.probesave_path), output_path=Path(args.probepkl_save_path), img_size=args.img_size, 
                                workers=args.n_workers, verbose=args.verbose, dataset=args.dataset)
    # 加载embs, 先固定名字是embeddings吧
    # load embslist
    # embslist是记录特征矩阵的dic,id:[]
    embsdic = {}
    embs_path = "{}{}.pkl".format(args.embspath, "embeddings")
    if not osp.exists(args.embspath):
        os.makedirs(args.embspath)
    if osp.exists(embs_path):
        print("========= Load Embs..... ==========")
        with open(embs_path,'rb') as f:	
            embsdic = pickle.load(f)	
            print(embsdic)
            print("========= Finish Load Embs..... ==========")

    # gait
    gait(args, embsdic)
    # save embslist
    pkl = open(embs_path, 'wb')
    pickle.dump(embsdic, pkl)
    print("========= Final Embs..... ==========")
    print(embsdic)

    # compare
    if args.comparegait:
        print("##############################comparegait")
        gaitcompare(args, embsdic, True)
    

                           
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)