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


root = os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
sys.path.append(root)
config = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))))) + "/PaddleSeg/contrib/PP-HumanSeg/"
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))))) + "/PaddleSeg/contrib/PP-HumanSeg/src")
print("#############################")
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))))) + "/PaddleSeg/contrib/PP-HumanSeg/src")
from seg_demo import seg_opengait_image
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))) + "/datasets")
from pretreatment import pretreat
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/models/")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
from modeling import models
# from baseline import Baseline

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="/home/jdy/Gaitdateset/Image/videos/001", help="path to images or video"
    )

    # 有剪影图 png的
    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--save_path", default="/home/jdy/Gaitdateset/Image01/videos/001", help="path to save"
    )
    # 只有剪影图
    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--savesil_path", default="/home/jdy/Gaitdateset/Image02/videos/001", help="path to save"
    )
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
    #gait
    parser.add_argument('--local_rank', type=int, default=0,
                        help="passed by torch.distributed.launch module")
    parser.add_argument('--phase', default='train',
                        choices=['train', 'test'], help="choose train or test phase")
    parser.add_argument('--log_to_file', action='store_true',
                        help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
    parser.add_argument('--iter', default=0, help="iter to restore")

    return parser


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


def imageflow_demo(predictor, vis_folder, current_time, args, id, type, seq):
    video_path = osp.join(args.path, id, type, seq)
    save_path = osp.join(args.save_path, id, type)
    savesil_path = osp.join(args.savesil_path, id, type)
    for video in glob.glob(video_path):
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

        while True:
            if frame_id == 10:
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
                
                            print("111111111111111111111111111111111111111111111")
                            save_name = "{}-{:03d}.png".format(seq.split('.')[0],frame_id)
                            print(final_config, save_name)

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

def gait(args):
    pkl_save = args.pkl_save_path

    print("========= Loading model..... ==========")
    # initialization(config_loader(cfgs["gaitmodel"]["cfg_path"]), False)
    gaitmodel, newcfgs = loadModel(**cfgs["gaitmodel"])
    gaitmodel.requires_grad_(False)
    gaitmodel.eval()
    print("========= Load Done.... ==========")

    loader = gaitmodel.test_loader
    for inputs in loader:
        ipts = gaitmodel.inputs_pretreament(inputs)
        print(ipts)
        retval, embs = gaitmodel.forward(ipts, args.whole_pkl_save_path)
        print(embs)
        del ipts


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
    
    data_dir = args.path
    save_dir = args.save_path
    for id in sorted(os.listdir(data_dir)):
        for type in sorted(os.listdir(os.path.join(data_dir,id))):
            for seq in sorted(os.listdir(os.path.join(data_dir,id,type))):
                print("############################111111")
                imageflow_demo(predictor, vis_folder, current_time, args, id, type, seq)
    

    print("###################")
    print(Path(args.pkl_save_path))
    print("###################")
    pretreat(input_path=Path(args.save_path), output_path=Path(args.pkl_save_path), img_size=args.img_size, workers=args.n_workers, verbose=args.verbose, dataset=args.dataset)
    pkl_save = args.pkl_save_path

    # gait
    gait(args)






                           
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)