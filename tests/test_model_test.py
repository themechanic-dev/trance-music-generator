import os

from tmg.jobs.kinds import model_test


def test_cache_info_counts_blobs_and_spots_partial_downloads(tmp_path):
    hub = tmp_path / "hub"
    repo = hub / "models--acme--tiny"
    (repo / "blobs").mkdir(parents=True)
    (repo / "snapshots" / "abc").mkdir(parents=True)
    blob = repo / "blobs" / "0123"
    with open(blob, "wb") as f:
        f.truncate(200 * 2**20)                                   # sparse 200 MB, no disk used
    os.symlink(blob, repo / "snapshots" / "abc" / "model.safetensors")
    assert model_test.cache_info("acme/tiny", hub) == (True, 0.2)   # the symlink is not counted twice
    assert model_test.cache_info("acme/missing", hub) == (False, 0.0)
    (repo / "blobs" / "4567.incomplete").write_bytes(b"x" * 10)
    downloaded, _size = model_test.cache_info("acme/tiny", hub)
    assert downloaded is False                                     # a download in progress is not a model


def test_summary_reads_like_a_status_line():
    ok = {"model": "stable-audio", "model_id": "stabilityai/stable-audio-open-1.0", "downloaded": True, "size_gb": 9.5,
          "access": "ok", "working": True, "load_s": 12.3, "seconds": 8.0, "gen_s": 8.9, "peak_vram_gb": 5.4, "rms_db": -14.2,
          "tested_utc": "2026-09-15T00:40:11+00:00", "wav": "/elsewhere/clip.wav"}
    text = model_test.summary(ok)
    assert text.splitlines()[0] == "stable-audio: stabilityai/stable-audio-open-1.0"
    assert "downloaded (9.5 GB in models/hf) · access ok" in text
    assert "WORKING: loaded in 12 s, 8 s clip in 9 s, peak VRAM 5.4 GB, level -14 dBFS" in text
    assert "tested 2026-09-15 00:40 UTC · /elsewhere/clip.wav" in text
    bad = {"model": "medium", "model_id": "facebook/musicgen-medium", "downloaded": False, "access": "gated - accept access",
           "working": False, "error": "RuntimeError: CUDA is not available"}
    text = model_test.summary(bad)
    assert "not downloaded · access gated - accept access" in text
    assert "NOT WORKING: RuntimeError: CUDA is not available" in text
    assert "tested" not in text
