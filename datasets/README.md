# Prepare datasets for Amodal Expander

Our Amodal-Expander augments the modal tracker [GTR](https://github.com/xingyizhou/GTR/tree/master) by training on [TAO-Amodal](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal). If interested, check [here](https://github.com/xingyizhou/GTR/blob/master/datasets/README.md) to see how GTR was trained.

## Download Datasets 
1. Download [TAO-Amodal](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal).
    ```bash
    git lfs install
    git clone git@hf.co:datasets/chengyenhsieh/TAO-Amodal
    ```

2. Download [Segment Object](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal-Segment-Object-Large). 
    > This dataset, collected from [LVIS](https://www.lvisdataset.org/) and [COCO](https://cocodataset.org/#home), is used in our [PasteNOcclude](https://github.com/WesleyHsieh0806/Amodal-Expander?tab=readme-ov-file#rabbit2-pastenocclude) augmentation technique.
    ```bash
    git lfs install
    git clone git@hf.co:datasets/chengyenhsieh/TAO-Amodal-Segment-Object-Large
    ```

3. Place or sim-link the downloaded datasets under `$Amodal-Expander/datasets/`. 
    You could refer to [sim-link.sh](./sim-link.sh) to generate symbolic link
    ```
    $Amodal-Expander/datasets/
        tao-amodal/
        segment-object-large/
        tao/
        lvis/
        coco/
    ```

<details><summary>Download TAO, LVIS and COCO (Optional)</summary>

* Download [TAO](https://taodataset.org/) dataset (Optional)
    TAO-Amodal shares the same sets of frames with TAO, so we only need to download the [annotations](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md).

* Download [LVIS](https://www.lvisdataset.org/) and [COCO](https://cocodataset.org/#home) (Optional)
    If you want to reproduce GTR or generate our Segment-Object dataset

</details>

</br>
Please follow the instructions below to pre-process individual datasets.
  <ul>
    <li>
      <a href="#tao-amodal">TAO-Amodal</a>
    </li>
    <li>
      <a href="#customized-dataset">Customized Dataset</a>
    </li>
    <li>
      <a href="#coco-and-lvis">COCO and LVIS (Optional)</a>
    </li>
    <ul>
        <li>
        <a href="#collection-details-of-segment-object">Collection Details of Segment-Object</a>
        </li>
    </ul>
    <li>
      <a href="#tao">TAO (Optional)</a>
    </li>
  </ul>

## TAO-Amodal
TAO-Amodal **does not** need to be further preprocessed as all the files are already provided in our dataset. 

```
   TAO-Amodal
    ├── frames
    │    └── train
    │       ├── ArgoVerse
    │       ├── BDD
    │       ├── Charades
    │       ├── HACS
    │       ├── LaSOT
    │       └── YFCC100M
    ├── amodal_annotations
    │    ├── train/validation/test.json
    │    ├── train_lvis_v1.json
    │    └── validation_lvis_v1.json
    ├── example_output
    │    └── prediction.json
    ├── BURST_annotations
    │    ├── train
    │         └── train_visibility.json
    │    ...
    
 ```


- <details>
    <summary>If interested, read the paragraphs below to check useful scripts to create variations of annotation formats.</summary>

    We used `train_lvis_v1.json` to train the Amodal Expander by viewing each image frame as independent sequences. `validation_lvis_v1.json` is used for [evaluation](https://github.com/WesleyHsieh0806/TAO-Amodal?tab=readme-ov-file#bar_chart-evaluation).
    
    `train_lvis_v1.json` was obtained through:
    ```bash
    python tools/create_tao_amodal_train_v1.py datasets/tao/amodal_annotations/train.json
    ```
    
    `validation_lvis_v1.json` was obtained through:
    ```bash
    python tools/create_tao_amodal_v1.py datasets/tao/amodal_annotations/validation.json 
    ```
    </detail>



- You can also check [MODEL_ZOO.md](../docs/MODEL_ZOO.md#inference-at-higher-fps) to create annotation JSON for running trackers at higher fps.

## Customized Dataset
1. To train/inference the model on your customized dataset, check detectron2 [tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#use-custom-datasets). 
2. Then, create a python script `your_dataset.py` to register your dataset following our [provided examples](../gtr/data/datasets/). 

3. Finally, import the dataset script (such as `your_dataset.py`) in [__init\__.py](./gtr/__init__.py), which enables the dataset to be loaded during training:
    ```python
    # Something like this
    from .data.datasets import your_dataset
    ```

## COCO and LVIS

Download COCO and LVIS data place them in the following way:

```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
coco/
    train2017/
    val2017/
    annotations/
        captions_train2017.json
        instances_train2017.json 
        instances_val2017.json
```

Next, prepare the merged annotation file using 

~~~
python tools/merge_lvis_coco.py
~~~

This creates `datasets/lvis/lvis_v1_train+coco_box.json` or `datasets/lvis/lvis_v1_train+coco_mask.json` (by setting `NO_SEG=False`)

#### Collection Details of [Segment-Object]((https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal-Segment-Object-Large)).

Check [tools/get_lvis_segmented_set.ipynb](../tools/get_lvis_segmented_set.ipynb) to see the dataset collection details.
After running the above notebook, `Segment-Object-Large` dataset will then be constructed like this under `datasets/lvis/`.
```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
    Segment-Object-Large/
        segment_object.json
        train2017/
```


## TAO

Download the data following the official [instructions](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md) and place them as 

```
tao/
    frames/
        val/
            ArgoVerse/
            AVA/
            BDD/
            Charades/
            HACS/
            LaSOT/
            TFCC100M/
        train/
            ArgoVerse/
            ...
        test/
            ArgoVerse/
            ...
    annotations/
        train.json
        validation.json
        test_without_annotations.json
```

Our model only uses the annotated frames ("keyframe"). To make the data management easier, we first copy the keyframes to a new folder

```
python tools/move_tao_keyframes.py --gt datasets/tao/annotations/validation.json --img_dir datasets/tao/frames --out_dir datasets/tao/keyframes
```

This creates `tao/keyframes/`

The TAO annotations are originally based on LVIS v0.5. We update them to LVIS v1 for validation.

```
python tools/create_tao_v1.py datasets/tao/annotations/validation.json
```

This creates `datasets/tao/annotations/validation_v1.json`.

For TAO test set, we'll convert the LVIS v1 labels back to v0.5 for the server-based test set evaluation.
