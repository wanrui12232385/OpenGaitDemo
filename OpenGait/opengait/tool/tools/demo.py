# %%
import torch
import numpy as np
import os
from glob import glob
from sklearn.linear_model import Lasso
from modeling import models
from utils import config_loader
from main import initialization
from tqdm import tqdm
import pickle
import json
import pandas as pd
from PIL import Image
import imageio
from sklearn.ensemble import RandomForestRegressor
# %%
torch.distributed.init_process_group('nccl', init_method='env://')
data_path = "/data/ydq/GaitClone/OUMVLP_Mesh/pkl/pkl_r64_mixori2_view"
id_list = sorted(os.listdir(data_path))[0:1]  # select id number
get_coef=False   # if True need use feature selection

print(id_list)

view_label_json_path = 'datasets/GaitClone/view_list.json'
view_json = json.load(open(view_label_json_path,'r'))
print(view_json)

# %%
### get model
def loadModel(model_type, cfg_path):
    Model = getattr(models, model_type)
    cfgs = config_loader(cfg_path)
    model = Model(cfgs, training=False)
    return model

cfgs = {  "main_encoder":{
    "model_type": "GaitGAN_AEV",
    "cfg_path": "config/stylegan/fixed/stylegan_aev_gaitclone.yaml"
},
}
print("========= Loading model..... ==========")
initialization(config_loader(cfgs["main_encoder"]["cfg_path"]), False)
main_encoder = loadModel(**cfgs["main_encoder"])
main_encoder.requires_grad_(False)
main_encoder.eval()
print("========= Load Done.... ==========")

# %%
## get w

def get_w(numpy_data):
    torch_data = torch.tensor(numpy_data[:,np.newaxis,:,:]).float().cuda()
    _, wp = main_encoder.main_encoder(torch_data, [])
    return wp.detach().cpu().numpy()

def save_func(tmp, data, ipts_type='image'):
    if ipts_type == 'image':
        for i, con in enumerate(data):
            im = Image.fromarray(con[0], mode='L')
            im.save(os.path.join(tmp, '%03d.png' % i))
    elif ipts_type == 'pkl':
        with open(os.path.join(tmp,'00.pkl'), 'wb') as f:
            pickle.dump(data[:,0,:,:], f)
    elif ipts_type == 'w':
        for i in range(len(data)):
            with open(os.path.join(tmp, str(i).zfill(2) + '.pkl'), 'wb') as f:
                pickle.dump(data[i], f)

def save_image(data, dataset, need='image', metric='euc'):
    root = '/data/ydq/GaitClone/output'
    # root = '/samba/ydq/GaitClone/output'
    # root = '/home/ydq/workspace/VG/tmp_dataset'
    # root = '/home1/data/ydq/GaitClone/output/'
    # root = '/home/ydq/workspace/VG/tmp_dataset'
    model = 'Lasso_test_y'
    # model = 'GaitGAN_AEV_triplet-10000'

    images, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views'] # n s c h w

    if "image" in need:
        root_path = os.path.join(root, dataset, model+'_image')
        os.makedirs(os.path.join(root_path),exist_ok=True)
        for i, id in enumerate(label[:]):
            tmp = os.path.join(root_path, str(id).zfill(5), str(seq_type[i]), str(view[i]))
            os.makedirs(tmp, exist_ok=True)
            save_func(tmp, images[i])
            save_gif(tmp, tmp, str(view[i]))

    if 'pkl' in need:
        root_path = os.path.join(root, dataset, model+'_pkl')
        os.makedirs(os.path.join(root_path),exist_ok=True)
        for i, id in enumerate(label[:]):        
            tmp = os.path.join(root_path, str(id).zfill(5), str(seq_type[i]), str(view[i]))
            os.makedirs(tmp, exist_ok=True)
            save_func(tmp, images[i], 'pkl')

    if 'w' in need:
        root_path = os.path.join(root, dataset, model+'_w')
        os.makedirs(os.path.join(root_path),exist_ok=True)
        for i, id in enumerate(label[:]):        
            tmp = os.path.join(root_path, str(id).zfill(5), str(seq_type[i]), str(view[i]))
            os.makedirs(tmp, exist_ok=True)
            save_func(tmp, data['w'], 'w')
    return

def save_gif(image_folder, save_folder, name="movie"):
    images = []
    filenames = sorted(glob(os.path.join(image_folder, '*.png')))
    # print(filenames)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(save_folder, f'{name}.gif'), images, fps=20)

def own_save(decoder, wp, labs, ty, vi):
    ori_stylegan_images = decoder(torch.tensor(wp).float().cuda(), content=None)[None,...]

    save_im = True
    need = 'image'
    if save_im:
        stylegan_images = (ori_stylegan_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        tmp = {
            'embeddings': stylegan_images.detach().cpu().numpy(),
            # 'embeddings': None,
            'labels': labs,
            'types': ty,
            'views': vi,
            # 'w': [render_style.detach().cpu().numpy(), ori_style.detach().cpu().numpy()]
        }
        save_image(tmp, 'OUMVLP', need=need)


view_label_list = []
pkl_data_list = []
for id_name in tqdm(id_list):
    view_paths = sorted(glob(os.path.join(data_path,id_name,'*/*')))
    for view_path in view_paths:
        pkl_path = sorted(glob(os.path.join(view_path, '*_*.pkl')))[0]
        pkl_data = pickle.load(open(pkl_path, 'rb')) / 127.5 - 1
        temp_wp = get_w(pkl_data)
        pkl_data_list.append(temp_wp)
        view_label_list.append(np.array(view_json[view_path.split('/')[-1]]).repeat(temp_wp.shape[0]))

all_w_data = np.concatenate(pkl_data_list,axis=0)
all_view_label = np.concatenate(view_label_list,axis=0)
if get_coef:
    view_w = all_w_data[:, :4, :].reshape(all_w_data.shape[0], -1)
    N, C = view_w.shape
    print(N,C)
    lasso = RandomForestRegressor()
    model_lasso = lasso.fit(view_w, all_view_label)
    coef = pd.Series(model_lasso.feature_importances_, index=range(C))
    coef.to_csv('debug_lasso_30.csv',index=False)
    coef_abs = coef
else:
    coef = pd.read_csv('debug_lasso_30.csv')
    # print(coef)
    coef_abs= coef.iloc[:,0]

coef_abs = coef_abs[coef_abs != 0].abs().sort_values(ascending = False)
coef_index = coef_abs.index
print(coef_index)
selected_num_start = 0
selected_num_end = 512
selected_view = pkl_data_list[5][:,:4,:].reshape(pkl_data_list[1].shape[0],-1)[:,coef_index[selected_num_start:selected_num_end]]
save_data = pkl_data_list[0].copy()
own_save(main_encoder.decoder, save_data, labs=[0],ty=['raw'],vi=['raw'])

tmp = pkl_data_list[0][:,:4,:].reshape(pkl_data_list[0].shape[0],-1)
tmp[:,coef_index[selected_num_start:selected_num_end]] = selected_view * 10
tmp = tmp.reshape(pkl_data_list[0][:,:4,:].shape)

# np.where(pkl_data_list[0][:,:4,:].reshape(pkl_data_list[0].shape[0],-1)[:,coef_index[:selected_num]],) = selected_view

changed_save_data = np.concatenate([tmp, pkl_data_list[0][:,4:,:]],axis=1)
own_save(main_encoder.decoder, changed_save_data, labs=[0],ty=[f'changed_{selected_num_start}_{selected_num_end}'],vi=[f'changed_{selected_num_start}_{selected_num_end}'])
print((save_data == changed_save_data).all())