import struct

from conftest import make_wav

from tmg.capture import pipewire

PW_DUMP = """[
 {"id": 57, "type": "PipeWire:Interface:Node", "info": {"props": {"media.class": "Audio/Sink",
   "node.name": "alsa_output.pci-0000_06_00.1.hdmi-stereo", "node.description": "GA106 HDMI"}}},
 {"id": 45, "type": "PipeWire:Interface:Node", "info": {"props": {"media.class": "Audio/Sink",
   "node.name": "alsa_output.pci-0000_08_00.3.iec958-stereo", "node.description": "Family 17h IEC958"}}},
 {"id": 46, "type": "PipeWire:Interface:Node", "info": {"props": {"media.class": "Audio/Source", "node.name": "cam"}}},
 {"id": 1, "type": "PipeWire:Interface:Client", "info": {"props": {}}}
]"""


def test_parse_default_sink():
    text = "update: id:0 key:'default.audio.sink' value:'{\"name\":\"alsa_output.pci-0000_06_00.1.hdmi-stereo\"}' type:'Spa:String:JSON'"
    assert pipewire.parse_default_sink(text) == "alsa_output.pci-0000_06_00.1.hdmi-stereo"
    assert pipewire.parse_default_sink("") is None


def test_parse_wpctl_inspect():
    text = 'id 57, type PipeWire:Interface:Node\n  * node.name = "alsa_output.pci-0000_06_00.1.hdmi-stereo"\n'
    assert pipewire.parse_wpctl_inspect(text) == "alsa_output.pci-0000_06_00.1.hdmi-stereo"


def test_parse_sinks():
    sinks = pipewire.parse_sinks(PW_DUMP)
    assert [s.id for s in sinks] == [57, 45]
    assert sinks[0].description == "GA106 HDMI"
    assert pipewire.parse_sinks("not json") == []


def test_repair_wav_header(tmp_path):
    path = make_wav(tmp_path / "a.wav", seconds=0.2)
    with open(path, "r+b") as f:  # zero the RIFF and data sizes, as an unfinished libsndfile header would have
        f.seek(4)
        f.write(struct.pack("<I", 0))
        f.seek(40)
        f.write(struct.pack("<I", 0))
    assert not pipewire.wav_is_valid(path)
    assert pipewire.repair_wav_header(path)
    assert pipewire.wav_is_valid(path)


def test_tail_level_db(tmp_path):
    path = make_wav(tmp_path / "b.wav", seconds=0.5, amplitude=0.5)
    rms, peak = pipewire.tail_level_db(path, 48000, 2)
    assert -6.5 < peak < -5.5          # 0.5 -> -6.02 dBFS
    assert -9.5 < rms < -8.5           # sine rms = peak - 3 dB
    silent = make_wav(tmp_path / "c.wav", seconds=0.5, amplitude=0.0)
    assert pipewire.tail_level_db(silent, 48000, 2) == (-100.0, -100.0)
