# Amodal Expander

   Official Training and Inference Code of Amodal Expander, Proposed in [Tracking Any Object Amodally](https://tao-amodal.github.io/).
  <!-- [**:paperclip: Paper Link**]() -->
   [**:orange_book: Project Page**](https://tao-amodal.github.io/) | [**:octocat: Official Github**](https://github.com/WesleyHsieh0806/TAO-Amodal)  | [**:paperclip: Paper Link**](https://arxiv.org/abs/2312.12433) | [**:pencil2: Citations**](#citations)



---

  Amodal Expander serves as a plug-in module that can “amodalize” any ex-isting detector or tracker with limited (amodal) training data.
   
   <div align="center">
  <a href="#school_satchel-get-started"><img width="95%" alt="TAO-Amodal" src="https://github.com/WesleyHsieh0806/Amodal-Expander/assets/55971907/70ddf677-2c88-40d4-8b15-b2b7825d4eff"></a>
   </div>


  :pushpin: Leave a :star: in our [official repository](https://github.com/WesleyHsieh0806/TAO-Amodal) to keep track of the updates.

---


  <h2> Table of Contents</h2>
  <ul>
    <li>
      <a href="#school_satchel-get-started">Get Started</a>
    </li>
    <li>
      <a href="#running-training-and-inference">Training and Inference</a>
    </li>
    <li>
      <a href="#bar_chart-evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#robot-demo">Demo</a>
    </li>
    <li>
      <a href="#rabbit2-pastenocclude">PasteNOcclude</a>
    </li>
    <li>
      <a href="#citations">Citations</a>
    </li>
  </ul>



---

## :school_satchel: Get Started

See [installation instructions](./docs/INSTALL.md).

## :running: Training and Inference
We augment the SOTA modal tracker [GTR](https://github.com/xingyizhou/GTR) with Amodal Expander by fine-tuning on [TAO-Amodal](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal) dataset.

Please [prepare datasets](./datasets/README.md) and check our [MODEL ZOO](./docs/MODEL_ZOO.md) for training/inference instructions.

## :bar_chart: Evaluation
After obtaining the prediction JSON `lvis_instances_results.json` through the above inference pipeline. You can evaluate the tracker results using our [evaluation toolkit](https://github.com/WesleyHsieh0806/TAO-Amodal?tab=readme-ov-file#bar_chart-evaluation).

## :robot: Demo
You can test our model on a single video through:
```bash
python demo.py --config-file configs/GTR_TAO_Amodal_Expander_PasteNOcclude.yaml \
               --video-input demo/input_video.mp4 \
               --output      demo/output.mp4 \
               --opts        MODEL.WEIGHTS /path/to/Amodal_Expander_PnO_45k.pth
```
> Use `--input video_folder/*.jpg` instead if the video consists of image frames.

## :rabbit2: PasteNOcclude
PasteNOcclude serves as a data augmentation technique to automatically generate more occlusion scenarios. 
Check the [Jupyter demo](./gtr/data/transforms/demo.ipynb) and implementation details ([link 1](./gtr/data/tao_amodal_dataset_modal_match_mapper.py#L274), [link 2](./gtr/data/transforms/paste_and_occlude_transform.py), [link 3](./gtr/data/transforms/paste_and_occlude_impl.py)).

<div align="center">
  <a href="https://tao-amodal.github.io/"><img width="95%" alt="TAO-Amodal" src="https://github.com/WesleyHsieh0806/Amodal-Expander/assets/55971907/c08286bf-3e8a-464e-b5e3-dd23f389962f"></a>
   </div>

## Acknowledgement
This repository is built upon [Global Tracking Transformer](https://github.com/xingyizhou/GTR) and [Detectron2](https://github.com/facebookresearch/detectron2).

## LICENSE
Check [here](https://github.com/xingyizhou/GTR?tab=readme-ov-file#license) for further details.

## Citations
``` bash
@article{hsieh2023tracking,
  title={Tracking any object amodally},
  author={Hsieh, Cheng-Yen and Khurana, Tarasha and Dave, Achal and Ramanan, Deva},
  journal={arXiv preprint arXiv:2312.12433},
  year={2023}
}
```
