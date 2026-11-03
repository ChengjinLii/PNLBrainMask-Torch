"""Offline-only reference runner. Execute in the author's TF 2.15 environment."""

import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def main():
    import numpy as np
    import tensorflow as tf
    import keras

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--capture", nargs="*", default=[])
    parser.add_argument("--volume", action="store_true")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    definition = json.loads((args.source / "CompNetBasicModel.json").read_text())
    if args.precision == "float64":
        # Diagnostic only: isolate layout/graph errors from float32 roundoff.
        keras.backend.set_floatx("float64")
        for layer in definition["config"]["layers"]:
            layer["config"]["dtype"] = "float64"
    model = keras.models.model_from_json(json.dumps(definition))
    matches = list(args.source.glob(f"weights-{args.view}-improvement-*.h5"))
    if len(matches) != 1:
        raise RuntimeError("Ambiguous reference weights")
    model.load_weights(matches[0])
    x = np.load(args.input, mmap_mode="r")
    if args.volume:
        # Pruning the two unused heads leaves the original final_op unchanged.
        model = keras.Model(model.input, model.get_layer("final_op").output)
        result = np.lib.format.open_memmap(args.output / "probability.npy", mode="w+", dtype=args.precision, shape=x.shape)
        for start in range(len(x)):
            result[start] = model(x[start:start + 1, ..., None], training=False).numpy()[0, ..., 0]
            if start % 32 == 0:
                print(f"TF {args.view}: {start}/256", flush=True)
        result.flush()
    else:
        probe = keras.Model(model.input, [model.get_layer(name).output for name in args.capture])
        result = probe(np.asarray(x)[..., None], training=False)
        np.savez(args.output / "tensors.npz", **{name: value.numpy() for name, value in zip(args.capture, result)})
    (args.output / "runtime.json").write_text(json.dumps({"tensorflow": tf.__version__, "keras": keras.__version__,
        "device": "cpu", "precision": args.precision, "onednn": False, "intra_threads": 2, "inter_threads": 1}, indent=2) + "\n")


if __name__ == "__main__":
    main()
