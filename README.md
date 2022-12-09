# OpenGaitDemo

配置环境
```
pip install -r requirements_demo.txt
```



运行demo
```
cd OpenGaitDemo
python3 OpenGait/opengait/tool/tools/track_gait.py video -f ByteTrack/exps/example/mot/yolox_x_mix_det.py -c ByteTrack/pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result  --phase test
```



### ByteTrack
可能会有yolox的问题
解决方案:
```

```

### PaddleSeg
