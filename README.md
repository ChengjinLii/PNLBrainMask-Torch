<div align="center">

# PNLBrainMask-Torch

**Diffusion MRI brain masking with pretrained PNL CompNet weights**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)](#dependencies)
[![Weights](https://img.shields.io/badge/Weights-PNL_v0.3-276A5D)](#pretrained-weights)

[Overview](#overview) | [Installation](#installation) | [Inputs](#inputs) | [Run](#run-brain-masking) | [Validation](VALIDATION.md) | [Citation](#citation) | [License](#license)

</div>

PNLBrainMask-Torch brings the [PNL CNN diffusion MRI brain segmentation
tool](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation) to PyTorch.
It uses the authors' pretrained three-view CompNet models, converted directly
from their released weights, and preserves the original masking workflow.

**3D b0 or 4D DWI in. Native-grid binary brain mask out.**

---

## Overview

```text
Mean b0
  -> rigid registration to the IIT 256^3 template
  -> 99th-percentile normalization
  -> sagittal / coronal / axial CompNet inference
  -> weighted probability fusion and thresholding
  -> inverse transform, hole filling, native-grid brain mask
```

The fusion weights are 0.1, 0.4 and 0.5; the threshold is strictly greater than
0.45. All three CompNet output heads are converted. Production inference uses
the first head, `final_op`, just as the original tool does.

This is a modified implementation maintained by Chengjin Li. The model and
method are credited to the original authors in [Citation](#citation).
The [validation record](VALIDATION.md) documents the conversion and measured
cross-framework differences.

---

## Installation

```bash
git clone https://github.com/ChengjinLii/PNLBrainMask-Torch.git
cd PNLBrainMask-Torch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

A compatible existing PyTorch environment can also be used. For a CPU-only
installation, install the CPU build before installing the package:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

### Dependencies

- Python >=3.10; PyTorch >=2.4,<3; NumPy >=1.24,<3; NiBabel >=5.2,<6.
- ANTs for rigid registration, inverse transformation and hole filling.
  Put `antsRegistrationSyNQuick.sh`, `antsRegistration`,
  `antsApplyTransforms`, `ImageMath` and `PrintHeader` on `PATH`, or pass `--ants-bin`.
- TensorFlow and Keras are used only for optional offline validation.
  HDF5 conversion uses optional `h5py`; neither is needed for inference.

The full native-grid pipeline is tested on Linux with ANTs 2.3.0. CNN-only
inference can run independently of ANTs.

---

## Pretrained Weights

The converted state dictionaries are included in [weights/](weights/).
Download the IIT template directly from NITRC:

```bash
pnl-brain-mask fetch --model-dir ./weights --accept-terms --template
```

The command verifies existing weights and downloads missing files. For a
wheel installation, choose any cache directory, such as `--model-dir ./models`.

Review [the PNL license](LICENSE) and the [IIT atlas terms](https://www.nitrc.org/projects/iit/)
before passing `--accept-terms`. The template is obtained from its original NITRC
server and is not bundled or mirrored here.

[weights/weights.json](weights/weights.json) records the original and converted
SHA256 hashes for every view. The same manifest is included in the installed
package. Files are checked before `torch.load(weights_only=True)`.

To reproduce the conversion:

```bash
python -m pip install '.[conversion]'
pnl-brain-mask fetch-author --model-dir ./author-models --accept-terms
pnl-brain-mask convert --source ./author-models --output ./converted-models
```

Original weights: [PNL v0.3 model archive](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/releases/download/v0.3/model_folder.tar.gz).
Archive SHA256: `c67210fd59e5bcaf8e0e730bdde9d8f28b817148a74d9586a39c9b6e86df5191`.
[IIT template download](https://www.nitrc.org/frs/download.php/11290/IITmean_b0_256.nii.gz)
SHA256: `2bb1a7bbe448702033a515e76164df6472828422ecef5cb4f996d91ec126dead`.

---

## Inputs

Provide a 3D b0 NIfTI, or a corrected 4D DWI NIfTI with matching b-values.
The 4D route averages measurements with `b <= 50` by default.
B-vectors are not used by this b0-based model.

The output is a binary int16 NIfTI on the input's native 3D voxel grid.
Nonfinite data and invalid baseline measurements produce explicit errors.

---

## Run Brain Masking

From a 3D b0:

```bash
pnl-brain-mask mask --input ./b0.nii.gz --output ./brain_mask.nii.gz \
  --model-dir ./weights --device cpu --threads 2 --record ./brain_mask.json
```

From a 4D DWI:

```bash
pnl-brain-mask mask --input ./dwi.nii.gz --bvals ./dwi.bval \
  --output ./brain_mask.nii.gz --model-dir ./weights \
  --b0-threshold 50 --ants-bin /path/to/ants/bin --device cpu --threads 2
```

Use `--template /path/IITmean_b0_256.nii.gz` to provide the template separately.
`--device auto` selects the first caller-visible CUDA GPU when available,
otherwise CPU. `--device cuda` requires a working CUDA runtime.
Default slice batch size is 1.

For CNN-only use with an already registered IIT-grid b0:

```python
import nibabel as nib
from pnl_brain_mask.inference import normalize, predict

registered = nib.load('registered_b0.nii.gz')  # IIT 256 x 256 x 256 grid
probabilities, fused, mask = predict(
    normalize(registered.get_fdata()), './weights', device='cpu', batch_size=1)
```

---

## Tests and Validation

```bash
python -m pip install '.[test]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

[VALIDATION.md](VALIDATION.md) contains the actual tensor, probability and mask
comparisons, reproduction commands and tested environment. Validation uses
local tests; this repository has no GitHub Actions workflows.

---

## Citation

Please cite the original software and methods:

- Palanivelu S, Cetin Karayumak S, Billah T, Zurrin R, Bouix S, Rathi Y.
  *CNN based diffusion MRI brain segmentation tool* (2020).
  [Code](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation),
  [DOI: 10.5281/zenodo.3665739](https://doi.org/10.5281/zenodo.3665739).
- Cetin-Karayumak S, Zhang F, Zurrin R, et al.
  *Harmonized diffusion MRI data and white matter measures from the Adolescent
  Brain Cognitive Development Study.* Scientific Data 11, 249 (2024).
  [DOI: 10.1038/s41597-024-03058-w](https://doi.org/10.1038/s41597-024-03058-w).
- Dey R, Hong Y. *CompNet: Complementary segmentation network for brain MRI
  extraction.* MICCAI (2018).
  [Architecture code](https://github.com/raun1/MICCAI2018---Complementary_Segmentation_Network-Raw-Code).
- Zhang S, Arfanakis K. *Evaluation of standardized and study-specific diffusion
  tensor imaging templates of the adult human brain.* NeuroImage 172, 40-50 (2018).

Identify this implementation and version when reporting use of the PyTorch port.
Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

---

## License

All or portions of this licensed product (such portions are the "Software") have
been obtained under license from The Brigham and Women's Hospital, Inc. and are
subject to the terms and conditions reproduced in [LICENSE](LICENSE).

The [original PNL software/data agreement](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/blob/d7fe0fd4976c2d5a3489526f75d2a00dae3e1fe1/LICENSE)
and its required notices are preserved in the repository and wheel.
The acquired model archive has no separate weight-specific license document;
the published project-wide agreement is retained with the converted weights.
IIT template permissions and dependency licenses remain separate.

Research use only. Review generated masks before quantitative analysis.
