# Amodal Expander Model Zoo

## Introduction

This file documents a collection of models reported in our paper.
Our experiments are trained on 4 RTX 3090 GPUs.


  <ul>
    <li>
      <a href="#model-list">Download Model</a>
    </li>
    <li>
      <a href="#tao-amodal">TAO-Amodal</a>
      <ul>
        <li>
        <a href="#training">Training</a>
        </li>
        <li>
        <a href="#inference">Inference</a>
        </li>
        <li>
        <a href="#inference-at-higher-fps">Inference at Higher FPS</a>
        </li>
        <li>
        <a href="#evaluation">Evaluation</a>
        </li>
      </ul>
    </li>
  </ul>

## Model List
If you don't have a specific preference for a particular model, we suggest downloading `Amodal Expander + PnO (45k iterations)`.

|         Name          | Model Weights |
|-----------------------|------------------|
|[Baseline (GTR)](../configs/GTR_TAO_DR2101.yaml) | [download](https://drive.google.com/file/d/1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD/view?usp=sharing) |
|[Amodal Expander](../configs/GTR_TAO_Amodal_Expander.yaml) | [download](https://huggingface.co/chengyenhsieh/Amodal-Expander/blob/main/Amodal_Expander_20k.pth) |
|[Amodal Expander + PnO](../configs/GTR_TAO_Amodal_Expander_PasteNOcclude.yaml) | [download](https://huggingface.co/chengyenhsieh/Amodal-Expander/blob/main/Amodal_Expander_PnO_20k.pth) |
|[Amodal Expander + PnO (45k iters)](../configs/GTR_TAO_Amodal_Expander_PasteNOcclude.yaml) | [download](https://huggingface.co/chengyenhsieh/Amodal-Expander/blob/main/Amodal_Expander_PnO_45k.pth) |

You can use the following command to download all the model weights of Amodal Expander.

```bash
git lfs install
git clone git@hf.co:chengyenhsieh/Amodal-Expander
```
<br>

#### How to Run the Model

The "Name" column contains a link to the config file. 
To train a model, run 

```
python train_net.py --num-gpus 4 --config-file /path/to/config/name.yaml
``` 

To evaluate a model with a trained/ pretrained model, run 

```
python train_net.py --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth
``` 

:warning: Make sure to customize `OUTPUT_DIR` in the config and place all model weights under `$Amodal-Expander/models/`.

<br>
<br>

## TAO-Amodal
### Training
1. Download [pre-trained GTR](https://drive.google.com/file/d/1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD/view?usp=sharing) and place the model weight under `$Amodal-Expander/models/`.

2. Training:

    * Training with PasteNOcclude
    ```bash
    python train_net.py --num-gpus 4 --config-file ./configs/GTR_TAO_Amodal_Expander_PasteNOcclude.yaml
    ```

    * Without PasteNOcclude

    ```bash
    python train_net.py --num-gpus 4 --config-file ./configs/GTR_TAO_Amodal_Expander.yaml
    ```

The training log and model weights will be saved in `OUTPUT_DIR`, specified in the `--config-file`.

<br>

### Inference

Specify `MODEL.WEIGHTS` and `OUTPUT_DIR`. 
> For example, `MODEL.WEIGHTS` could be `./models/Amodal_Expander_PnO_45k.pth`.

```bash
python train_net.py \
    --config-file ./configs/GTR_TAO_Amodal_Expander.yaml \
    --eval-only \
    MODEL.WEIGHTS /path/to/model/weights \
    OUTPUT_DIR /path/to/output/folder
```

This creates `inference_tao_amodal_val_v1/lvis_instances_results.json` in the `OUTPUT_DIR`.

<br>

### Inference at Higher FPS
You can also run trackers on TAO-Amodal at higher fps by including non-annotated frames into the annotation JSON.

1. Create new annotation JSON
    ```bash
    python tools/create_30fps_json.py \
        --annotations datasets/tao_amodal/amodal_annotations/validation_lvis_v1.json \
        --images-dir /path/to/TAO-Amodal/frames \
        --fps 5
    ```
    This outputs `validation_lvis_v1_5fps.json` in `datasets/tao_amodal/amodal_annotations/`.

2. Register the dataset `tao_amodal_v1_5fps` in [tao_amodal.py](../gtr/data/datasets/tao_amodal.py#L183)

3. Create a new config ([example](../configs/GTR_TAO_Amodal_Expander_PasteNOcclude.yaml)) and changes its value of `DATASETS.TEST` to `tao_amodal_v1_5fps`. Use this new config file to run inference.

4. Remove predictions at non-annotated frames before evaluation.
    ```bash
    python tools/rm_30fps_dt_to_1fps.py \
        --annotations datasets/tao_amodal/amodal_annotations/validation_lvis_v1_5fps.json \
        --track-results path/to/5fps/lvis_instances_results.json
    ```

This creates `path/to/5fps/lvis_instances_results_to_1fps.json` in your `OUTPUT_FOLDER`.

<br>

### Evaluation
After running the inference, you can evaluate the tracking results with the generated JSON (`lvis_instances_results.json`). Please refer to our [eval toolkit](https://github.com/WesleyHsieh0806/TAO-Amodal?tab=readme-ov-file#bar_chart-evaluation) for further details.
