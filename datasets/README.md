# Prepare datasets for Amodal Expander

Our Amodal-Expander augments the modal tracker [GTR](https://github.com/xingyizhou/GTR/tree/master) by training on [TAO-Amodal](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal). If interested, check [here](https://github.com/xingyizhou/GTR/blob/master/datasets/README.md) to see how GTR was trained.

## Download Datasets 
1. Download [TAO-Amodal](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal).

2. Download [Segment Object](). 
    > This dataset, collected from [LVIS](https://www.lvisdataset.org/) and [COCO](https://cocodataset.org/#home), is used in our [PasteNOcclude](https://github.com/WesleyHsieh0806/Amodal-Expander?tab=readme-ov-file#rabbit2-pastenocclude) augmentation technique.


3. Sim-link the downloaded datasets under `$Amodal-Expander/datasets/`. 
    You could refer to [sim-link.sh](./sim-link.sh) to generate symbolic link
    ```
    $Amodal-Expander/datasets/
        tao-amodal/
        segment-object-large/
        tao/
        lvis/
        coco/
    ```

4. Download [TAO](https://taodataset.org/) dataset (Optional)
    TAO-Amodal shares the same sets of frames with TAO, so we only need to download the [annotations](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md).

5. Download [LVIS](https://www.lvisdataset.org/) and [COCO](https://cocodataset.org/#home) (Optional)
    If you want to reproduce GTR or generate our Segment-Object dataset


Now, please follow the instructions below to pre-process individual datasets.

## TAO-Amodal

Register TAO-Amodal into the DatasetCatalog by using tools gtr/data/datasets/tao_amodal.py

### COCO and LVIS (Optional)

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

#### Create TAO-Amodal-Segment-Object Dataset from LVIS for PasteAndOcclude Training.
This is a prerequisite if you want to fine-tune GTR with PasteAndOcclude.
To use the data augmentation technique, PasteAndOcclude, to fine-tune your model, you need to further prepare a set of segmented objects from `datasets/lvis/lvis_v1_train+coco_mask.json`.


This could be done by running [tools/get_lvis_segmented_set.ipynb](../tools/get_lvis_segmented_set.ipynb).
You can also directly download the TAO-Amodal-Segment-Object Dataset [here](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal-Segment-Object).

The dataset `Segment-Object` will then be structured like this under `datasets/lvis/Segment-Object`.
```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
    Segment-Object/
        segment_object.json
        train2017/
```


### TAO (Optional)

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

Since we use detectron2 to load the dataset during training, testing. Remember to run gtr/data/datasets/tao.py to register tao into DatasetCatalog in detectron2

