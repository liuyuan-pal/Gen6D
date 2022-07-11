# Gen6D

Gen6D is able to estimate 6DoF poses for unseen objects like the following video.

![](assets/example.gif)

## [Project page](https://liuyuan-pal.github.io/Gen6D/) | [Paper](https://arxiv.org/abs/2204.10776)

## Todo List

- [x] Pretrained models and evaluation codes.
- [x] Pose estimation on custom objects.
- [ ] Training codes.

## Installation

Required packages are list in `requirements.txt`. 

## Download

1. Download pretrained models, GenMOP dataset and processed LINEMOD dataset at [here](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/yuanly_connect_hku_hk/EkWESLayIVdEov4YlVrRShQBkOVTJwgK0bjF7chFg2GrBg?e=Y8UpXu).
2. Organize files like
```
Gen6D
|-- data
    |-- model
        |-- detector_pretrain
            |-- model_best.pth
        |-- selector_pretrain
            |-- model_best.pth
        |-- refiner_pretrain
            |-- model_best.pth
    |-- GenMOP
        |-- chair 
            ...
    |-- LINEMOD
        |-- cat 
            ...
```

## Evaluation


```shell
# Evaluate on the object TFormer from the GenMOP dataset
python eval.py --cfg configs/gen6d_pretrain.yaml --object_name genmop/tformer

# Evaluate on the object cat from the LINEMOD dataset
python eval.py --cfg configs/gen6d_pretrain.yaml --object_name linemod/cat
```

Metrics about ADD-0.1d and Prj-5 will be printed on the screen.

### Qualitative results

3D bounding boxes of estimated poses will be saved in `data/vis_final/gen6d_pretrain/genmop/tformer`.
Ground-truth is drawn in green while prediction is drawn in blue.

![](assets/results.jpg)

Intermediate results about detection, viewpoint selection and pose refinement will be saved in `data/vis_inter/gen6d_pretrain/genmop/tformer`.

![](assets/detection.jpg)

This image shows detection results.


![](assets/selection.jpg)

This image shows viewpoint selection results.
The first row shows the input image to the selector. 
The second row shows the input images rotated by the estimated in-plane rotation (left column) or the ground-truth in-plane rotation(right column)
Subsequent 5 rows show the predicted (left) or ground-truth (right) 5 reference images with nearest viewpoints to the input image.

![](assets/refinement.jpg)

This image shows the pose refinement process.
The red bbox represents the input pose, the green one represents the ground-truth and the blue one represents the output pose for the current refinement step. 

## Pose estimation on custom objects

Please refer to [custom_object.md](custom_object.md)

## Acknowledgements
In this repository, we have used codes or datasets from the following repositories. 
We thank all the authors for sharing great codes or datasets.

- [PVNet](https://github.com/zju3dv/pvnet)
- [hloc](https://github.com/cvg/Hierarchical-Localization)
- [COLMAP](https://github.com/colmap/colmap)
- [Google Scanned Objects](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects)
- [MVSNet_pl](https://github.com/kwea123/MVSNet_pl)
- [AnnotationTools](https://github.com/luigivieira/Facial-Landmarks-Annotation-Tool)

## Citation
```
@inproceedings{liu2022gen6d,
  title={Gen6D: Generalizable Model-Free 6-DoF Object Pose Estimation from RGB Images},
  author={Liu, Yuan and Wen, Yilin and Peng, Sida and Lin, Cheng and Long, Xiaoxiao and Komura, Taku and Wang, Wenping},
  booktitle={ECCV},
  year={2022}
}
```