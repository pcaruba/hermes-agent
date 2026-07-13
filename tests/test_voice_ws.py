import math
import struct

import pytest

from gateway.platforms.voice_ws import (
    MAX_UTTERANCE_BYTES,
    VoiceProtocolError,
    VoiceSession,
    pop_complete_sentences,
    rms_amplitude,
)


def test_sentence_split_retains_incomplete_tail():
    sentences, tail = pop_complete_sentences("Hello there. How are")
    assert sentences == ["Hello there."]
    assert tail == "How are"


def test_sentence_split_flushes_tail():
    sentences, tail = pop_complete_sentences("First! Last fragment", flush=True)
    assert sentences == ["First!", "Last fragment"]
    assert tail == ""


def test_rms_amplitude_is_normalised():
    silence = b"\x00\x00" * 160
    loud = struct.pack("<160h", *([16_384] * 160))
    assert rms_amplitude(silence) == 0.0
    assert rms_amplitude(loud) == pytest.approx(0.5, rel=0.01)


class _FakeWs:
    def __init__(self):
        self.text = []
        self.binary = []
        self.closed = False

    async def send_str(self, value):
        self.text.append(value)

    async def send_bytes(self, value):
        self.binary.append(value)


@pytest.mark.asyncio
async def test_audio_limit_is_enforced():
    session = VoiceSession(ws=_FakeWs(), adapter=object(), session_id="s", session_key=None)
    session.audio.extend(b"x" * MAX_UTTERANCE_BYTES)
    with pytest.raises(VoiceProtocolError, match="60 second"):
        await session.accept_audio(b"xx")


@pytest.mark.asyncio
async def test_barge_in_clears_audio_and_emits_listening():
    ws = _FakeWs()
    session = VoiceSession(ws=ws, adapter=object(), session_id="s", session_key=None)
    session.turn_id = "turn_1"
    session.audio.extend(b"1234")
    await session.barge_in()
    assert session.audio == b""
    assert any('"type": "barge_in_ack"' in frame for frame in ws.text)
    assert any('"state": "listening"' in frame for frame in ws.text)
