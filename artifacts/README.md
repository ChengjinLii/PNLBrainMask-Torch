# Installable Wheel

`pnl_brain_mask-0.1.0-py3-none-any.whl` is built from this repository with
`python -m build --wheel`. It contains the Python package, architecture,
weight checksum manifest and PNL licenses. The large model state dictionaries
are downloaded separately by `pnl-brain-mask fetch`.

SHA256: `44f16ebe938db2ad4be0a5d4f1fe834b73ffdcdcdc1205c7ba965781a5916dac`.

SlicerDSpace pins both the repository commit and these wheel bytes. This small
artifact avoids downloading all weights twice during managed environment setup.
It was built with Python 3.11, setuptools 84.2.1 and wheel 0.48.0.
