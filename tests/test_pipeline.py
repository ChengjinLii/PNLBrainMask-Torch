import nibabel as nib
import numpy as np
import pytest

from pnl_brain_mask.pipeline import ants_workspace, mean_b0, run_ants
from pnl_brain_mask.weights import acquire_author_weights


def test_mean_b0_selection_and_clipping(tmp_path):
    data = np.zeros((3, 4, 5, 3), dtype=np.float32)
    data[..., 0], data[..., 1], data[..., 2] = -2, 8, 1000
    values = tmp_path / "bvals"
    np.savetxt(values, [0, 50, 1000])
    result = mean_b0(nib.Nifti1Image(data, np.eye(4)), values)
    np.testing.assert_array_equal(result, np.full((3, 4, 5), 3.0))
    with pytest.raises(ValueError, match="requires"):
        mean_b0(nib.Nifti1Image(data, np.eye(4)))
    np.savetxt(values, [1000, 1000, 1000])
    with pytest.raises(ValueError, match="No b0"):
        mean_b0(nib.Nifti1Image(data, np.eye(4)), values)


def test_no_download_before_terms(tmp_path):
    destination = tmp_path / "absent"
    with pytest.raises(PermissionError):
        acquire_author_weights(destination)
    assert not destination.exists()


def test_ants_argv_is_shell_free_and_thread_bounded(tmp_path, monkeypatch):
    from pnl_brain_mask import pipeline
    executable = tmp_path / "ants tools" / "antsApplyTransforms"
    executable.parent.mkdir()
    executable.touch()
    executable.chmod(0o755)
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    run_ants(["antsApplyTransforms", "-i", "input with spaces.nii.gz"], ants_bin=executable.parent, threads=2)
    args, kwargs = calls[0]
    assert args[0] == [str(executable), "-i", "input with spaces.nii.gz"]
    assert kwargs["env"]["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == "2"
    assert not kwargs.get("shell")


def test_path_only_ants_sets_antspath(tmp_path, monkeypatch):
    from pnl_brain_mask import pipeline
    script = tmp_path / "antsRegistrationSyNQuick.sh"
    script.touch()
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("ANTSPATH", raising=False)
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **kw: calls.append(kw))
    run_ants([script.name, "-d", "3"])
    assert calls[0]["env"]["ANTSPATH"] == str(tmp_path)


def test_ants_private_aliases_handle_external_paths_with_spaces(tmp_path):
    original = tmp_path / "original tools"
    original.mkdir()
    for command in ("antsRegistrationSyNQuick.sh", "antsRegistration", "antsApplyTransforms", "ImageMath", "PrintHeader"):
        (original / command).touch()
        (original / command).chmod(0o755)
    template = tmp_path / "template with spaces.nii.gz"
    template.touch()
    work = tmp_path / "private"
    work.mkdir()
    tools, reference = ants_workspace(work, template, original)
    assert (tools / "PrintHeader").resolve() == original / "PrintHeader"
    assert reference.resolve() == template
    assert " " not in str(tools) and " " not in str(reference)


def test_ants_rejects_whitespace_workspace(tmp_path):
    with pytest.raises(ValueError, match="work_dir"):
        ants_workspace(tmp_path / "work with spaces", tmp_path / "template")
