# OV_DETR

## Introduction


OV_DETR is a open-vocabulary object detection model based on transformer. We reproduced the model of the paper.


## Model Zoo

| Backbone | Model | Images/GPU  | Inf time (fps) | Box AP | Box AP Seen| Box AP Unseen| Config |
|:------:|:--------:|:--------:|:--------------:|:------:|:------:|:------:|:------:|
| R-50 | OV_DETR  | -- | -- | -- | -- | -- |[config](ov_detr_r50_1x_coco.yml) |

## Prepare
1. Download the open-vocabulary [Annotations](https://bj.bcebos.com/v1/paddledet/data/coco/zero-shot.zip), replace the relevant path in the configuration file
```
ov_detr_r50_1x_coco.yml
  ....
  text_embedding: zeroshot_w.npy
  ....
  clip_feat_path: clip_feat_coco.pkl


_base_/ov_detr_coco_detection.yml
  TrainDataset:
    ....
    anno_path: instances_train2017_seen_2_proposal.json

  EvalDataset:
    ....
    anno_path: instances_val2017_all.json
    ....
```
2. [Multi-scale deformable attention custom OP compilation](../../ppdet/modeling/transformers/ext_op/README.md)
## Note
   paddlepadle>=2.4.2

   cuda >=10.2

## GPU multi-card training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ov_detr/ov_detr_r50_1x_coco.yml --fleet --eval
```

## Export model for inference
```bash
#Due to the different mechanisms of training and reasoning, it is necessary to uninstall the custom operator before exporting the model
pip uninstall deformable-detr-op

#export model
python tools/export_model.py -c configs/ov_detr/ov_detr_r50_1x_coco.yml  -o weights=best.pdparams

#inference
python deploy/python/infer.py --model_dir=./output_inference/ov_detr_r50_1x_coco --image_file=./demo/000000014439.jpg --device=GPU

```

## Citations
```
@InProceedings{zang2022open,
 author = {Zang, Yuhang and Li, Wei and Zhou, Kaiyang and Huang, Chen and Loy, Chen Change},
 title = {Open-Vocabulary DETR with Conditional Matching},
 journal = {European Conference on Computer Vision},
 year = {2022}
}
```
