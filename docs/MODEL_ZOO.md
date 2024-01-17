# Amodal Expander model zoo

## Introduction

This file documents a collection of models reported in our paper.
Our experiments are trained on 4 RTX 3090 GPUs.

#### How to Read the Tables

The "Name" column contains a link to the config file. 
To train a model, run 

```
python train_net.py --num-gpus 4 --config-file /path/to/config/name.yaml
``` 

To evaluate a model with a trained/ pretrained model, run 

```
python train_net.py --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth
``` 

|         Name          | Model Weights |
|-----------------------|------------------|
|[Baseline](../configs/GTR_TAO_DR2101.yaml) | [download](https://drive.google.com/file/d/1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD/view?usp=sharing) |
|[Amodal Expander](../configs/GTR_TAO_Amodal_Expander.yaml) | [download](https://drive.google.com/file/d/1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD/view?usp=sharing) |
|[Amodal Expander + PnO](../configs/GTR_TAO_DR2101.yaml) | [download](https://drive.google.com/file/d/1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD/view?usp=sharing) |
|[Amodal Expander + PnO (45k iters)](../configs/GTR_TAO_DR2101.yaml) | [download](https://drive.google.com/file/d/1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD/view?usp=sharing) |






|             |          Grouping           ||
First Header  | Second Header | Third Header |
 ------------ | :-----------: | -----------: |
Content       |          *Long Cell*        ||
Content       |   **Cell**    |         Cell |

New section   |     More      |         Data |
And more      | With an escaped '\|'         ||  
[Prototype table]


## Overview of TAO-Amodal Training/Inference

1. Structure and downlowded datasets as described [here](https://github.com/xingyizhou/GTR/blob/master/datasets/README.md#coco-and-lvis)
    1. After downloading lvis and tao dataset, you could [generate Sim-Link](https://github.com/WesleyHsieh0806/GTR/blob/a43b207ea8d284c4cac87eaa8489ba8f55ab70b2/datasets/sim-link.sh) by running:

    ``` bash
    cd datasets
    bash sim-link.sh
    ```

    2. Create a registration file such as gtr/data/datasets/tao.py(tao-amodal.py) to register tao/tao-amodal into DatasetCatalog, then import this script in [gtr init file](https://github.com/WesleyHsieh0806/GTR/blob/master/gtr/__init__.py)

5. Modify validation annotation with LVIS v1 as described [here](https://github.com/xingyizhou/GTR/blob/master/datasets/README.md#tao)
    1. only category ids are modified here
    2. Run the following command:

    ```bash
    python tools/create_tao_v1.py datasets/tao/annotations/validation_with_freeform.json  # TAO
    python tools/create_tao_amodal_v1.py datasets/tao/amodal_annotations/validation_with_freeform_amodal_boxes_Aug10_2022.json  # Tao-amodal
    python tools/create_tao_amodal_train_v1.py datasets/tao/amodal_annotations/train_with_freeform_amodal_boxes_may12_2022_oof_visibility.json  # Create Tao-amodal Training json
    ```

    This creates *datasets/tao/annotations/validation_with_freeform_v1.json* and *datasets/tao/amodal_annotations/validation_with_freeform_amodal_boxes_Aug10_2022_v1.json*
    
6. Run evaluation on TAO and TAO-Amodal
    1. Modify the evaluation dataset in the [config file](GTR/configs/GTR_TAO_DR2101.yaml)
    2. Refer to the details for [TAO evaluation](https://github.com/WesleyHsieh0806/GTR/blob/master/docs/MODEL_ZOO.md#evaluation-on-tao) and [TAO-Amodal evaluation](https://github.com/WesleyHsieh0806/GTR/blob/master/docs/MODEL_ZOO.md#evaluation-on-tao-amodal).

### Folder Structure
```

-- GTR
|   |- train_net.py
|   |- Readme.md
|   |- datasets
|   |   |- tao
|   |   |- lvis
|   |-- environment_setup.sh
```
## TAO

|         Name          |   validation mAP |  Test mAP | Download |
|-----------------------|------------------|-----------|----------|
|[GTR_TAO_DR2101](../configs/GTR_TAO_DR2101.yaml) | 22.5  | 20.1 | [model](https://drive.google.com/file/d/1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD/view?usp=sharing) |

#### Note

- The model is evaluated on TAO keyframes only, which are sampled in ~1 frame-per-second.
- Our model is trained on LVIS+COCO only. The TAO training set is not used anywhere.
- Our model is finetuned on a detection-only CenterNet2 model trained on LVIS+COCO ([config](./configs/C2_LVISCOCO_DR2101_4x.yaml), [model](https://drive.google.com/file/d/1WCrfbyNhMryB4ryV5piLG3NLgU3pUvcz/view?usp=sharing)). Download or train the model and place it as `GTR_ROOT/models/C2_LVISCOCO_DR2101_4x.pth` before training. Training the detection-only models takes ~3 days on 8 GPUs.
- Training GTR takes ~13 hours on 4 V100 GPUs (32G memory).

#### Evaluation on TAO
1. Off-the-shelf evaluation:
    * Download the model [weight](https://drive.google.com/file/d/1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD/view?usp=sharing)
    * Modify evaluation dataset in the [config file](../configs/GTR_TAO_DR2101.yaml)
    * Use the following command
    ```bash
    cd ..
    python train_net.py --config-file /home/chengyeh/TAO-Amodal/GTR/GTR/configs/GTR_TAO_DR2101.yaml --eval-only MODEL.WEIGHTS /data3/chengyeh/checkpoints/GTR/GTR_TAO_DR2101.pth
    ```

## TAO-Amodal
#### Training on LVIS+COCO
We propose the PasteAndOcclude Training to adapt GTR into an amodal tracker for TAO-Amodal evaluation.

|         Name          |   validation mAP |  Test mAP | Download |
|-----------------------|------------------|-----------|----------|
|[GTR_TAO_Amodal_DR2101](../configs/GTR_TAO_Amodal_DR2101.yaml) |  | | [model]() |

You will first need to obtain a set of segmented objects from LVIS+COCO as described [here](../datasets/README.md#prepare-lvis-segmented-set-for-pasteandocclude-training).

Training from GTR baseline model:
```bash
python train_net.py --num-gpus 4 --config-file ./configs/GTR_TAO_Amodal_Training_DR2101.yaml

# or use this to resume training
python train_net.py --num-gpus 4 --config-file ./configs/GTR_TAO_Amodal_Training_DR2101.yaml --resume MODEL.WEIGHTS [Path/to/Model/weights]

e.g.,
python train_net.py --num-gpus 4 --config-file ./configs/GTR_TAO_Amodal_Training_DR2101.yaml --resume MODEL.WEIGHTS /data3/chengyeh/TAO-Amodal-experiments/GTR/PasteAndOcclude-3Segments/GTR_TAO_Amodal_Training_DR2101/model_final.pth OUTPUT_DIR /data3/chengyeh/TAO-Amodal-experiments/GTR/PasteAndOcclude-3Segments/GTR_TAO_Amodal_Training_DR2101/ INPUT.VIDEO.PASTE_AND_OCCLUDE.NUM_SEGMENTS 3 SOLVER.MAX_ITER 60000
```

Training from the pretrained detector:
```bash
python train_net.py --num-gpus 4 --config-file ./configs/GTR_TAO_DR2101_PasteNOcclude.yaml

# or use this to resume training
python train_net.py --num-gpus 4 --config-file ./configs/GTR_TAO_DR2101_PasteNOcclude.yaml --resume MODEL.WEIGHTS [Path/to/Model/weights]
```

Training Amodal detector from scratch
```bash
python train_net.py --num-gpus 4 --config-file ./configs/C2_LVISCOCO_DR2101_4x_PasteNOcclude.yaml

# Modal Detector
python train_net.py --num-gpus 4 --config-file ./configs/C2_LVISCOCO_DR2101_4x.yaml
```

Training Amodal Expander from pre-trained GTR:
```bash
python train_net.py --num-gpus 4 --config-file ./configs/GTR_TAO_Amodal_Expander.yaml

python train_net.py --num-gpus 4 --config-file ./configs/GTR_COCO_Amodal_Expander.yaml

# with PasteNOcclude
python train_net.py --num-gpus 4 --config-file ./configs/GTR_TAO_Amodal_Expander_PasteNOcclude.yaml
python train_net.py --num-gpus 4 --config-file ./configs/GTR_TAO_Temporal_Amodal_Expander_PasteNOcclude.yaml
```

Fine-tune the Regression Head of GTR on TAO-Amodal-Train:
```bash

```

##### Note

- This model is finetuned on the GTR trained on LVIS+COCO ([config](../configs/GTR_TAO_DR2101.yaml), [model](https://drive.google.com/file/d/1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD/view?usp=sharing)). Download or train the model and place it as `GTR_ROOT/models/GTR_TAO_DR2101.pth` before training.

#### Evaluation on TAO-Amodal
1. Off-the-shelf evaluation:
    * Download the model [weight](https://drive.google.com/file/d/1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD/view?usp=sharing)
    * Modify evaluation dataset in the [config file](../configs/GTR_TAO_Amodal.yaml)
    * Modify .INPUT.NOT_CLAMP_BOX in the [CenterNet Config](../third_party/CenterNet2/centernet/config.py), the training boxes annotations are clamped during [transform](../gtr/data/gtr_dataset_mapper.py), and the predicted boxes are clamped [here](https://github.com/WesleyHsieh0806/GTR/blob/7138b95b5c7951e763af2a3ced15cb29ac8fc9de/gtr/modeling/roi_heads/custom_fast_rcnn.py#L172).
    * Use the following command
    
```bash
cd ..
python train_net.py --config-file /home/chengyeh/TAO-Amodal-Root/TAO-GTR/configs/GTR_TAO_Amodal.yaml --eval-only MODEL.WEIGHTS /data3/chengyeh/checkpoints/GTR/GTR_TAO_DR2101.pth

```

2. 30 fps evaluation:
    * use the following command

```bash
# run on 30 fps images
CUDA_VISIBLE_DEVICES=7 python train_net.py --config-file /home/chengyeh/TAO-Amodal-Root/TAO-GTR/configs/GTR_TAO_Amodal_30fps.yaml --eval-only MODEL.WEIGHTS /data3/chengyeh/checkpoints/GTR/GTR_TAO_DR2101.pth
```

    * remove dt results on non annotated images (refer to TAO-Amodal/tools/rm_30fps_dt_to_1fps.py)
    * re-evaluate the output json after removing results on non-annotated images
    * Due to unknown reasons, the official implementation might generate tracks assigned with multiple categories. In this case, we could remove those tracks before evaluation using:
```bash
python tools/post_process_track_result.py --track-result /compute/trinity-2-1/chengyeh/lvis_instances_results.json --output-result /data3/chengyeh/TAO-Amodal-experiments/GTR/15fps/GTR_TAO_Amodal_15fps/inference_tao_amodal_val_v1_15fps/lvis_instances_results_post_processed.json
```
3. GTR+PasteAndOcclude Evaluation:

```bash
cd ..
python train_net.py --config-file /home/chengyeh/TAO-Amodal-Root/TAO-GTR/configs/GTR_TAO_Amodal.yaml --eval-only MODEL.WEIGHTS /data3/chengyeh/TAO-experiments/GTR/GTR_TAO_Amodal_Training_DR2101/model_final.pth OUTPUT_DIR /data3/chengyeh/TAO-Amodal-experiments/GTR/GTR_TAO_Amodal_Training_DR2101/    
```
4. Out-of-frame stats:
    You can obtain some statistics of the out-of-frame predictions for your model using `tools/get_oof_prediction_stats.ipynb`. 
    Do so by modifying the `annotations` and `predictions` in the first cell.

5. AmodalExpander Eval:
```bash
cd ..
python train_net.py --config-file /home/chengyeh/TAO-Amodal-Root/TAO-GTR/configs/GTR_TAO_Amodal_Expander.yaml --eval-only MODEL.WEIGHTS /data3/chengyeh/TAO-Amodal-experiments/GTR/AmodalExpander/TAO-Amodal/GTR_TAO_Amodal_Expander/model_final.pth OUTPUT_DIR /data3/chengyeh/TAO-Amodal-experiments/GTR/AmodalExpander/TAO-Amodal/GTR_TAO_Amodal_Expander
```
#### Inference at Higher FPS
You can also run trackers on TAO-Amodal at higher fps by including non-annotated frames into the annotation JSON.

1. Create new annotation JSON
    ```bash
    python tools/create_30fps_json.py \
        --annotations datasets/tao_amodal/amodal_annotations/validation_lvis_v1.json --images-dir /path/to/TAO-Amodal/frames \
        --fps 5
    ```
    This outputs `validation_lvis_v1_5fps.json` in `datasets/tao_amodal/amodal_annotations/`.

2. Register the dataset `tao_amodal_v1_5fps` in [tao_amodal.py](../gtr/data/datasets/tao_amodal.py#L183)

3. Create a new [config](../configs/GTR_TAO_Amodal_Expander_PasteNOcclude.yaml) and changes the value of `DATASETS.TEST` to `tao_amodal_v1_5fps`. Use this config for inference.

4. Remove predictions at non-annotated frames before evaluation.
    ```bash
    python tools/rm_30fps_dt_to_1fps.py --annotations datasets/tao_amodal/amodal_annotations/validation_lvis_v1_5fps.json --track-results path/to/your/5fps/lvis_instances_results.json
    ```

#### Temporal Amodal Expander with Kalman Filter
After generating a tracking result json file using the above evaluation pipeline, we further apply Kalman filter on the tracking results in an online manner to make the Amodal Expander temporal-aware. 


#### Evaluation Tracking Results with TAO-Amodal Toolkit
Please refer to [TAO-Amodal](https://github.com/WesleyHsieh0806/TAO-Amodal) repository about the usage of evaluation toolkit.
