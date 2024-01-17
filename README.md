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
First [prepare datasets](./datasets/README.md) and check our [MODEL ZOO](./docs/MODEL_ZOO.md) for training/inference instructions.


## :rabbit2: PasteNOcclude
PasteNOcclude serves as a data augmentation technique to automatically generate more occlusion scenarios. 
Check the [Jupyter demo]() and implementation details ([link 1](), [link 2]()).

<div align="center">
  <a href="https://tao-amodal.github.io/"><img width="95%" alt="TAO-Amodal" src="https://github.com/WesleyHsieh0806/Amodal-Expander/assets/55971907/c08286bf-3e8a-464e-b5e3-dd23f389962f"></a>
   </div>

## Acknowledgement
This repository is built upon [Global Tracking Transformer](https://github.com/xingyizhou/GTR) and [Detectron2](https://github.com/facebookresearch/detectron2).

## LICENSE
Check [here](https://github.com/xingyizhou/GTR?tab=readme-ov-file#license) for further details.

## Citations
``` bash
@misc{hsieh2023tracking,
    title={Tracking Any Object Amodally},
    author={Cheng-Yen Hsieh and Tarasha Khurana and Achal Dave and Deva Ramanan},
    year={2023},
    eprint={2312.12433},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
