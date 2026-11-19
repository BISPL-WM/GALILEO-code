# Third-party code

| Files | Source | License |
|---|---|---|
| `galileo/models/{lvsm,transformer,camera}.py`, `galileo/pos_enc/rayrope.py`, `galileo/trainer.py` (structure) | [RayRoPE](https://github.com/wuyu-wy/RayRoPE) (Yu Wu) | MIT (see `LICENSE`) |
| `galileo/pos_enc/prope.py` | PRoPE reference implementation, "Cameras as Relative Positional Encoding" ([arXiv 2507.10496](https://arxiv.org/abs/2507.10496)) | MIT (header of the file) |
| `galileo/pos_enc/urope.py` | [URoPE](https://github.com/Applied-Intuition-Open-Source/URoPE) (Applied Intuition, Inc.) | Apache-2.0 (header of the file); modified, changes listed in the header |
| `galileo/video_metrics.py` (`features`, `jedi`) | JEDi's V-JEPA feature and MMD recipe ([oooolga/JEDi](https://github.com/oooolga/JEDi)), reimplemented | Apache-2.0 (JEDi) |
| `galileo/losses/perceptual.py` | [Crowdsampling the Plenoptic Function](https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function) (via RayRoPE) | see the source repository |

Data and models used by the scripts, not redistributed here: the Stereo4D annotations (gs://stereo4d), the rectified
Stereo4D videos (Hugging Face `KevinMathew/stereo4d-{left,right}eye-perspective`, non-commercial research use), the
ARROW model (`ifradlin/arrow`), torchvision's RAFT weights, and the MatConvNet VGG-19 weights.

Used by the video metrics, not redistributed here: CoTracker3 (`facebookresearch/co-tracker`, loaded through torch.hub; CC BY-NC 4.0), the V-JEPA ViT-H/16 weights and SSv2 probe (`facebookresearch/jepa`; CC BY-NC 4.0) with the model code of the `vjepa` package, and torchvision's RAFT weights.
