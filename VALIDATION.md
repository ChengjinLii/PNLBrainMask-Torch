# Conversion and Validation

Local validation performed on 2026-11-04. The original weights were converted
without retraining. The numerical record includes both successful checks and
the strict FP32 tolerance failures.

## Source and Conversion

The reference is PNL commit `d7fe0fd4976c2d5a3489526f75d2a00dae3e1fe1`,
with the [official v0.3 model archive](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation/releases/download/v0.3/model_folder.tar.gz).
Archive and individual-file SHA256 values were checked against the acquired
author files; [weights.json](weights/weights.json) contains the full provenance.
The architecture JSON was copied unchanged, SHA256
`989fd5a64d9307268441c3525d59b31d59c20db864f0262ef53ad53ec495bdae`.

The graph has 177 layers: 57 Conv2D, 15 Conv2DTranspose, 54 BatchNormalization,
23 Dropout, 16 Concatenate, 10 MaxPooling2D, one Add and one InputLayer.
Each view has 18,705,959 parameters and 414 PyTorch state tensors.

The converter maps HWIO to OIHW, and transposed-convolution HWOI to IOHW,
without a spatial flip. BatchNorm gamma, beta, moving mean, moving variance and
epsilon (0.001) are preserved. All three output branches are present.
There are 360 learned/statistical source tensors per view; **all 1,080 tensors
across three views were checked element-for-element and match exactly**.
The other 54 state tensors per view are new zero-valued PyTorch BatchNorm counters.

Production execution prunes the unused ancestors of `xfinal_op` and
`res_1_final_op`, retaining the original `final_op`. The validation checks
pruned and full-graph `final_op` for exact equality on each slice test case.

## Environment and Inputs

- Linux x86_64, Python 3.11, PyTorch 2.5.1+cpu, NumPy 1.26.4.
- Reference TensorFlow 2.15.1 and Keras 2.15.0 in a separate interpreter.
- CPU only, two Torch/TF intra-op threads, one TF inter-op thread.
  TensorFlow oneDNN optimizations disabled for the reference.
- ANTs 2.3.0 for the shared registration and both native-space cleanups.
- Public IIT mean-b0 template for initial slice tests; HCP subject 100307's real
  mean b0 for full-volume pairing, after one rigid registration to the IIT grid.
- Image data are not published in this repository. Input/transform hashes are
  recorded in the numerical artifacts and local preparation receipt.

Both frameworks receive the same float32-normalized volume and slice axes.
Sharing a transform isolates framework conversion from stochastic registration
variation; it is not a claim that separate registration runs are identical.

## Actual Results

Three views, zero inputs, seeded random inputs and anatomical slices 64/128/192
were checked at 13 layer checkpoints. The checkpoints cover encoder, pooling,
BatchNorm, bottleneck, transposed convolution, skip concatenation, the two
complementary heads and the reconstruction head.

| Check | Measured result |
| --- | --- |
| Author weight tensors | 1,080/1,080 exactly equal |
| FP64 diagnostic, all checkpoints | Maximum absolute error `4.902744876744691e-13` |
| FP64 diagnostic, output heads | Maximum absolute error `2.7677860003905153e-13` |
| FP32 sagittal full-volume probability | Max `2.2214651107788086e-4`, mean `6.721319056879338e-9` |
| FP32 coronal full-volume probability | Max `1.1029839515686035e-4`, mean `2.205714499936562e-9` |
| FP32 axial full-volume probability | Max `8.219480514526367e-5`, mean `3.6377221235284196e-9` |
| FP32 fused probability | Max `4.412233829498291e-5`, mean `3.2805373972877305e-9` |
| IIT-grid fused binary mask | 0 different / 16,777,216 voxels; Dice 1.0 |
| Native-grid final binary mask | 0 different / 3,658,350 voxels; Dice 1.0 |
| Native brain voxel count | 697,887 in both frameworks |
| Native geometry | Identical shape and affine within `atol=1e-4` |

Intermediate FP32 checks use `atol=1e-3, rtol=1e-4`. Strict output/probability
checks use `atol=1e-5, rtol=1e-4`. Some output heads and full-volume probabilities
**fail the latter criterion**. This is retained as `passed: false` in the
[FP32 report](validation/hcp100307-fp32.json); the validation command correctly
returns nonzero. The criterion has not been silently relaxed.

An additional FP64 diagnostic casts the same author weights to float64 in both
frameworks, changing only arithmetic precision. All 117 checkpoint arrays pass
`atol=1e-9, rtol=1e-9`. Together with the exact weight audit this supports the
correctness of layout and graph mapping and is consistent with accumulated
FP32 rounding differences. It does **not** make production FP32 bitwise equal.

The original pinned author's actual `normalize`, `split`, `multi_view_fast`,
`ANTS_inverse_transform`, `npy_to_nifti` and `save_nifti` functions were also
executed independently. Normalization, view restoration, fusion mask and native
cleanup outputs match the port element-for-element on this case.
The default linear inverse interpolation, ImageMath FillHoles and subsequent
uint8 cast are intentionally preserved; optional `maskfilter` is not enabled.

A separate full native-grid run through the installed Torch CLI completed
successfully. Unit tests also cover input validation, device policy, graph
pruning, checksums before loading and operation without TF/Keras/HDF5 imports.

The plugin adapter was also run on the original 4D HCP DWI: 18 b0 measurements,
finite/nonempty binary native-grid output, 698,176 brain voxels, followed by
verified cache reuse. This **separate registration run is an integration smoke
test**, not the paired native-contract mask above. Its distinct voxel count is
not a framework-pairing result. See [plugin smoke record](validation/plugin-smoke-initial.json).

After resumption, 20 standalone tests pass, including complete author-tensor
audits and ANTs path handling. The NumPy 1.x normalization result was rechecked
against the pinned author function after adding explicit float32 division for
NumPy 2 compatibility. Original FP32 reports and strict thresholds are unchanged.
ANTs registration uses the original shell script's default one registration
thread; the configured two-thread budget applies to CNN inference and the other
ANTs operations. The preparation receipt records the requested budget.

Raw records:
[initial IIT FP32](validation/iit-fp32-initial.json),
[HCP FP32](validation/hcp100307-fp32.json),
[HCP FP64 diagnostic](validation/hcp100307-fp64-diagnostic.json),
[native scientific contract](validation/hcp100307-native-contract.json).

## Reproduction

Install the package with `.[test]` and acquire the original weights using the
README conversion commands. Run the exhaustive weight audit:

```bash
PNL_AUTHOR_MODELS=/path/to/author-models OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 python -m pytest -q
```

Prepare a shared registered input:

```bash
python tools/prepare_case.py --input ./dwi.nii.gz --bvals ./dwi.bval \
  --template ./IITmean_b0_256.nii.gz --ants-bin /path/to/ants/bin \
  --output .work/prepared
```

Use a separate TF 2.15 environment for the reference. Optional `.[validation]`
is available for constructing it, but is not a production dependency.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python tools/validate_pair.py \
  --tf-python /path/to/tf215/bin/python --source ./author-models \
  --model-dir ./weights --volume .work/prepared/rigid-Warped.nii.gz \
  --output .work/paired --full-volume

# Diagnostic, not production inference:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python tools/validate_pair.py \
  --tf-python /path/to/tf215/bin/python --source ./author-models \
  --model-dir ./weights --volume .work/prepared/rigid-Warped.nii.gz \
  --output .work/fp64 --precision float64

python tools/check_contract.py --script /path/to/pinned/pipeline/dwi_masking.py \
  --prepared .work/prepared --paired .work/paired \
  --ants-bin /path/to/ants/bin --output .work/native-pair
```

The FP32 command's strict failure does not prevent writing the probability and
mask artifacts needed for the native check. Repeated serialization can produce
different `.pt` bytes with different PyTorch versions; the source tensor audit
is separate from the byte-level download checks.

## Installation and Integration Acceptance

The published v0.1.0 implementation is commit
`8d201a50447ebea8b10a4f252bfd262931603970`. The managed installer pins the
33,462-byte wheel with SHA256
`44f16ebe938db2ad4be0a5d4f1fe834b73ffdcdcdc1205c7ba965781a5916dac`.
The environment was created from scratch using the actual installer, not a mock:
Python 3.11.18, ANTs 2.3.0, PyTorch 2.5.1+cpu and NumPy 2.4.6.
ANTs image roundtrip, all three views/all three heads, installation receipt,
and a second verified reuse passed. TensorFlow, Keras, h5py and conversion are
absent. See [managed installation](validation/managed-installation.json).

The current optional `pnl-cnn` profile was also downloaded publicly and verified:
three `.pt` files, weight manifest, full license and the original NITRC IIT
template. No legacy HDF5 was selected or downloaded. Historical HDF5 records
remain only in the explicit `pnl-cnn-legacy` profile. Reuse with the downloader
disabled passed. See [public resources](validation/public-optional-resources.json).

Cross-environment reuse was tested with network access disabled: a compatible
existing Torch environment retained its framework and copied the checked shared
Torch assets, then passed model loading and three-head forward checks. The
managed route does not require running the optional resource installer first;
it fetches missing Torch assets directly, or reuses checked optional assets.
See [cache reuse](validation/cross-environment-cache.json).

Final local regressions: **20 standalone tests passed** and **201 PNL-facing
plugin/resource/environment/distribution tests passed**, with no errors, failures
or skips in the final run (CMake 3.31.6 supplied on PATH). The plugin test set is
not a claim of full Slicer SDK/GUI qualification. Original strict FP32 failures
remain unchanged and separate from these regression results.
See [regression record](validation/plugin-regressions.json).

The real 4D HCP DWI was run again through the plugin using that freshly installed
managed Python, model directory and ANTs binaries. Native-grid, binary, finite,
nonempty output and verified cache reuse passed. This remains a separate
registration/integration run, not a probability-pairing result.
See [managed plugin smoke](validation/managed-plugin-smoke.json).

## Scope

The mask agreement above is for one real subject with a common registration.
CUDA/MPS parity, other ANTs versions, larger slice batches, Windows/macOS full
pipelines, and multi-subject segmentation accuracy have not been established.
No new clinical validation or anatomical-quality claim follows from these checks.
Inspect brain boundaries, cerebellum and brainstem before quantitative use.

Compared with the original input handling, this implementation rejects nonfinite
input and nonpositive normalization scale. The standalone CLI accepts NIfTI;
NRRD/NHDR conversion, browser QC and training are outside its scope.
