# TESN: Transformers Enhanced Segmentation Network
## Introduction
__We proposed Transformers Enhanced Segmentation Network (TESN) which made the following improvements to the original Mask R-CNN to improve the accuracy of instance segmentation:<br>
a) TESN adopt multi-stage architecture to provide high-quality target detection.<br>
b) TESN Introduce Transformer into the mask head to provide high-quality segmentation.__<br>
<br>
<br>
![TESN](https://github.com/Wzl-98/TESN/raw/main/Fig/Mask_head.png)<br>
__The framework of the mask head. (a) mask head of original Mask R-CNN. (b) mask head of TESN. (c) convolution and up-sampling. (d) Transformer layer.__<br>
<br>
<br>
__Results on MS COCO dataset :__<br>
![Results](https://github.com/Wzl-98/TESN/raw/main/Fig/Results.png)<br>
## Prepared to train
### Environment
__python 3.7.10<br>
cuda 10.1<br>
torch 1.5.0<br>
mmcv-full 1.3.8<br>
mmdet 2.18.0__
### File
__1)Place fcn_mask_transformer_head.py and transformer.py under ../mmdetection/mmdet/models/roi_heads/mask_heads<br>
2)Place TESN_COCO.py under ../mmdetection<br>
3)add 'from .fcn_mask_transformer_head import FCNMaskTransformerHead' into ../mmdetection/mmdet/models/roi_heads/mask_heads/__init__.py__<br>
## Train
__Run 'python tools/train.py TESN_COCO.py'__
## Test
__Run 'python tools/test.py TESN_COCO.py'__
## Acknowledgement
__With greatly appreciation for [open-mmlab](https://github.com/open-mmlab) for providing [mmdetection](https://github.com/open-mmlab/mmdetection) source code.__
