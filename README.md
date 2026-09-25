
<p align="center">
  <h1 align="center"><strong>DreamMimic: Learning Visuomotor Whole-Body Loco-Manipulation via World Model</strong></h1>
  <p align="center">
    <br>
    <strong>IROS 2026 (Accepted)</strong>
    <br>
  </p>
</p>

<p align="center">
  <a href="https://dreammimic.github.io/">
    <img src="https://img.shields.io/badge/Project-Website-green?style=flat&logo=googlechrome&logoColor=white">
  </a>
  <a href="https://arxiv.org/abs/2608.22278">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat&logo=arxiv&logoColor=white">
  </a>
</p>

## 🏠 Overview

DreamMimic is a world-model-assisted visual policy distillation framework for
contact-rich humanoid loco-manipulation. The codebase focuses on distilling a
privileged teacher policy into a deployable vision-based student policy.

Core features:

- Predictive world-model supervision for stable long-horizon distillation.
- Interaction-aware prediction heads and PCG-style teacher guidance.
- End-to-end scripts for teacher testing and student train/test/eval.

## 🧬 DreamMimic vs InterMimic

> [!NOTE]
> DreamMimic is directly built on top of the **InterMimic** teacher backbone.
> In this repository, `dreammimic/` provides the core simulator/task framework,
> while `dm_scripts/` exposes the DreamMimic release entrypoints for training,
> testing, and evaluation.
>
> For the upstream base implementation, see
> [InterMimic](https://github.com/Sirui-Xu/InterMimic).

## 📖 Getting Started

### 1) Environment

You can use either the pinned environment file or the manual Isaac Gym setup.

Option A (recommended, reproducible):

```bash
conda env create -f requirements.yaml
conda activate dreammimic
```

Option B (manual, aligned with InterMimic-style setup):

```bash
conda create -n dreammimic python=3.8
conda activate dreammimic
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

Install [Isaac Gym (Preview 4)](https://developer.nvidia.com/isaac-gym)
following NVIDIA's official instructions.

After activating conda, export library path before running gym scripts:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### 2) Dataset preparation

Download and place datasets as follows:

- OMOMO processed motions:
  [Google Drive](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view?usp=sharing)
  -> unzip to `InterAct/OMOMO_new/`
- Distillation reference data (teacher retarget/correction):
  [Google Drive](https://drive.google.com/file/d/1l2E5qR97Ap8jrLrJPHmtNT8DDW1qKhY_/view?usp=sharing)
  -> place under repository data layout as needed by your chosen config.

Notes:

- Some configs use `InterAct/OMOMO_new`, while others use
  `InterAct/OMOMO` + `InterAct/OMOMO_retarget`. Keep both prepared if you run
  multiple experiment variants.
- BEHAVE-specific runs expect files under `InterAct/behave/`.

### 3) Checkpoints

- Teacher checkpoints:
  [Google Drive](https://drive.google.com/drive/folders/1biDUmde-h66vUW4npp8FVo2w0wOcK2_k?usp=sharing)
- Student checkpoint (example):
  [Google Drive](https://drive.google.com/file/d/1GNFOjBRmiIIxYtfnG9WvK4fELKnDWroR/view?usp=sharing)

## 🚀 Commands

Primary scripts are in `dm_scripts/`:

### Data replay

```bash
bash dm_scripts/data_replay_omomo.sh
```

### Teacher (OMOMO)

Test:

```bash
bash dm_scripts/test_teacher_omomo.sh [CHECKPOINT_PATH] [NUM_ENVS]
```

Eval:

```bash
bash dm_scripts/eval_teacher_omomo.sh [CHECKPOINT_PATH] [NUM_ENVS] [OUTPUT_DIR]
```

Train (if needed):

```bash
bash dm_scripts/train_teacher_omomo.sh [NUM_ENVS] [OUTPUT_DIR]
```

### Student (DreamMimic main pipeline)

Train:

```bash
bash dm_scripts/train_student_dreammimic.sh [NUM_ENVS] [OUTPUT_DIR]
```

Test:

```bash
bash dm_scripts/test_student_dreammimic.sh [CHECKPOINT_PATH] [NUM_ENVS] [OUTPUT_DIR]
```

Eval:

```bash
bash dm_scripts/eval_student_dreammimic.sh [CHECKPOINT_PATH] [NUM_ENVS] [OUTPUT_DIR] [RECORD_VIDEO]
```

## 🗂️ dm_scripts Organization

Release entrypoints currently maintained in `dm_scripts/`:

- `common.sh`
- `data_replay_omomo.sh`
- `train_teacher_omomo.sh`
- `test_teacher_omomo.sh`
- `eval_teacher_omomo.sh`
- `train_student_dreammimic.sh`
- `test_student_dreammimic.sh`
- `eval_student_dreammimic.sh`

Student wrappers now use shortened cfg names under `dreammimic/data/cfg/`:
`dm_ref_train.yaml`, `dm_ref_test.yaml`, `dm_ref_eval.yaml`, and train cfg
`train/rlg/dm_ref_pcg.yaml`.

### Checkpoint and Output Defaults

- Default checkpoints are resolved from `ckpts/`.
- If a default file is missing in `ckpts/`, the script auto-copies from the
  legacy `checkpoints/` location (same filename, release-safe alias path).
- `test` / `eval` scripts default outputs to `/tmp/dreammimic_test/*` or
  `/tmp/dreammimic_eval/*` to avoid creating new `checkpoints/...` run folders.

## 🔗 Citation

If you find DreamMimic useful, please cite:

```bibtex
@inproceedings{yin2026dreammimic,
  title={DreamMimic: Learning Visuomotor Whole-Body Loco-Manipulation via World Model},
  author={Yin, Jie and Lai, Xingyu},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026},
  eprint={2608.22278},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2608.22278}
}
```

This IROS 2026 DreamMimic release is based on the InterMimic codebase and
extends it for world-model-assisted visuomotor distillation.

## 👏 Acknowledgements

DreamMimic relies on and extends multiple open-source projects, including:

- [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)
- [rl_games](https://github.com/Denys88/rl_games)
- [PHC](https://github.com/ZhengyiLuo/PHC)
- [InterMimic](https://github.com/Sirui-Xu/InterMimic)
- [InterAct](https://github.com/wzyabcas/InterAct)
- [OMOMO](https://github.com/lijiaman/omomo_release)
- [BEHAVE](https://github.com/xiexh20/BEHAVE)

Please follow the license and usage terms of each dependency and dataset.
