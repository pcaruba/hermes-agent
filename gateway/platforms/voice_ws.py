"""Authenticated, full-duplex voice session for the Hermes API server.

The websocket lives on the same aiohttp listener and Bearer trust boundary as
/v1/chat/completions. Android sends 16 kHz mono PCM16 frames as binary messages.
A JSON control message with type=commit_audio finalises the utterance; the
server transcribes it, runs Eva, emits text deltas, synthesises complete
sentences, and sends Opus/Ogg audio as binary frames.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import struct
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import logging

logger = logging.getLogger("voice_ws")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] voice_ws: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

import math
import re
import struct
import tempfile
import time
import uuid
import wave

VOICE_SAMPLE_RATE = 16_000
VOICE_CHANNELS = 1
VOICE_SAMPLE_WIDTH = 2
MAX_UTTERANCE_BYTES = VOICE_SAMPLE_RATE * VOICE_SAMPLE_WIDTH * 60
SENTENCE_RE = re.compile(r"^(.+?[.!?](?:[\"')\]]+)?(?:\s+|$)|.+?\n+)", re.S)


class VoiceProtocolError(ValueError):
    pass


def rms_amplitude(pcm16: bytes) -> float:
    """Normalised RMS for little-endian mono PCM16, used for telemetry/tests."""
    if len(pcm16) < 2:
        return 0.0
    usable = pcm16[: len(pcm16) - (len(pcm16) % 2)]
    samples = struct.unpack(f"<{len(usable) // 2}h", usable)
    if not samples:
        return 0.0
    mean_square = sum(float(s) * float(s) for s in samples) / len(samples)
    return min(1.0, math.sqrt(mean_square) / 32768.0)


def pop_complete_sentences(buffer: str, *, flush: bool = False) -> tuple[List[str], str]:
    """Pop sentence/newline-delimited units while retaining an incomplete tail."""
    sentences: List[str] = []
    rest = buffer
    while rest:
        match = SENTENCE_RE.match(rest)
        if not match:
            break
        sentence = match.group(1).strip()
        rest = rest[match.end() :]
        if sentence:
            sentences.append(sentence)
    if flush and rest.strip():
        sentences.append(rest.strip())
        rest = ""
    return sentences, rest


class VoiceSession:
    """One persistent Android voice conversation over one websocket."""

    def __init__(
        self,
        *,
        ws: Any,
        adapter: Any,
        session_id: str,
        session_key: Optional[str],
    ) -> None:
        self.ws = ws
        self.adapter = adapter
        self.session_id = session_id
        self.session_key = session_key or session_id
        self.audio = bytearray()
        self.history: List[Dict[str, str]] = []
        self.turn_id = ""
        self._agent_ref: List[Any] = [None]
        self._turn_task: Optional[asyncio.Task] = None
        self._tts_tasks: set[asyncio.Task] = set()
        self._tts_order_lock = asyncio.Lock()
        self._cancelled_turns: set[str] = set()
        self._sentence_index = 0
        self._text_buffer = ""
        self._send_lock = asyncio.Lock()

    async def send_json(self, frame_type: str, **payload: Any) -> None:
        if self.ws.closed:
            return
        frame = {"type": frame_type, "session_id": self.session_id, **payload}
        async with self._send_lock:
            if not self.ws.closed:
                await self.ws.send_str(json.dumps(frame, ensure_ascii=False))

    async def send_state(self, state: str, *, turn_id: str = "") -> None:
        await self.send_json("state", state=state, turn_id=turn_id or self.turn_id)

    async def accept_audio(self, data: bytes) -> None:
        if not data:
            return
        if not self.audio:
            logger.info("voice audio started session=%s bytes=%d", self.session_id, len(data))
        if len(self.audio) + len(data) > MAX_UTTERANCE_BYTES:
            raise VoiceProtocolError("utterance exceeds 60 second limit")
        self.audio.extend(data)

    async def commit_audio(self) -> None:
        if self._turn_task and not self._turn_task.done():
            raise VoiceProtocolError("a voice turn is already running")
        if not self.audio:
            raise VoiceProtocolError("no audio buffered")
        pcm = bytes(self.audio)
        self.audio.clear()
        self.turn_id = f"turn_{uuid.uuid4().hex}"
        logger.info(
            "voice audio committed session=%s turn=%s bytes=%d duration_ms=%d rms=%.4f",
            self.session_id,
            self.turn_id,
            len(pcm),
            int(len(pcm) / (VOICE_SAMPLE_RATE * VOICE_SAMPLE_WIDTH) * 1000),
            rms_amplitude(pcm),
        )
        self._cancelled_turns.discard(self.turn_id)
        self._turn_task = asyncio.create_task(self._process_turn(pcm, self.turn_id))

    async def barge_in(self, *, acknowledge: bool = True) -> None:
        turn = self.turn_id
        if turn:
            self._cancelled_turns.add(turn)
        agent = self._agent_ref[0]
        if agent is not None:
            try:
                agent.interrupt("Voice barge-in")
            except Exception:
                logger.debug("voice agent interrupt failed", exc_info=True)
        for task in list(self._tts_tasks):
            task.cancel()
        self._tts_tasks.clear()
        self.audio.clear()
        if acknowledge and not self.ws.closed:
            await self.send_json("barge_in_ack", turn_id=turn)
            await self.send_state("listening", turn_id=turn)

    async def close(self) -> None:
        await self.barge_in(acknowledge=False)
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _process_turn(self, pcm: bytes, turn_id: str) -> None:
        try:
            await self.send_state("thinking", turn_id=turn_id)
            transcript = await asyncio.to_thread(self._transcribe_pcm, pcm)
            logger.info(
                "voice transcription complete session=%s turn=%s chars=%d",
                self.session_id, turn_id, len(transcript),
            )
            if turn_id in self._cancelled_turns:
                return
            if not transcript:
                await self.send_json("error", code="no_speech", message="No speech detected", turn_id=turn_id)
                await self.send_state("listening", turn_id=turn_id)
                return
            await self.send_json("final_transcript", text=transcript, turn_id=turn_id)
            self.history.append({"role": "user", "content": transcript})

            loop = asyncio.get_running_loop()
            self._text_buffer = ""
            self._sentence_index = 0

            def on_delta(delta: Optional[str]) -> None:
                if not delta or turn_id in self._cancelled_turns:
                    return
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._handle_delta(delta, turn_id))
                )

            def on_tool_progress(event_type: str, tool_name: str = "", preview: str = "", args=None, **kwargs: Any) -> None:
                if turn_id in self._cancelled_turns:
                    return
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.send_json(
                        "activity", turn_id=turn_id, event=event_type,
                        tool=tool_name or "", description=preview or tool_name or event_type,
                    ))
                )

            def on_tool_start(tool_call_id: str, function_name: str, function_args: Any) -> None:
                on_tool_progress("tool.started", function_name, f"Using {function_name}", function_args)

            def on_tool_complete(tool_call_id: str, function_name: str, function_args: Any, function_result: Any) -> None:
                on_tool_progress("tool.completed", function_name, f"Finished {function_name}", function_args)

            result, _usage = await self.adapter._run_agent(
                user_message=transcript,
                conversation_history=list(self.history[:-1]),
                session_id=self.session_id,
                stream_delta_callback=on_delta,
                tool_progress_callback=on_tool_progress,
                tool_start_callback=on_tool_start,
                tool_complete_callback=on_tool_complete,
                agent_ref=self._agent_ref,
                gateway_session_key=self.session_key,
            )
            if turn_id in self._cancelled_turns:
                return
            final_text = result.get("final_response", "") if isinstance(result, dict) else ""
            await self._flush_text(turn_id)
            if self._tts_tasks:
                await asyncio.gather(*list(self._tts_tasks), return_exceptions=True)
            if final_text:
                self.history.append({"role": "assistant", "content": final_text})
            await self.send_json("response_complete", text=final_text, turn_id=turn_id)
            await self.send_state("listening", turn_id=turn_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("voice turn failed session=%s turn=%s", self.session_id, turn_id)
            await self.send_json("error", code="voice_turn_failed", message=str(exc)[:300], turn_id=turn_id)
            await self.send_state("listening", turn_id=turn_id)
        finally:
            self._agent_ref[0] = None

    async def _handle_delta(self, delta: str, turn_id: str) -> None:
        if turn_id in self._cancelled_turns:
            return
        await self.send_json("response_text_delta", delta=delta, turn_id=turn_id)
        self._text_buffer += delta
        sentences, self._text_buffer = pop_complete_sentences(self._text_buffer)
        for sentence in sentences:
            self._queue_tts(sentence, turn_id)

    async def _flush_text(self, turn_id: str) -> None:
        sentences, self._text_buffer = pop_complete_sentences(self._text_buffer, flush=True)
        for sentence in sentences:
            self._queue_tts(sentence, turn_id)

    def _queue_tts(self, sentence: str, turn_id: str) -> None:
        index = self._sentence_index
        self._sentence_index += 1
        task = asyncio.create_task(self._synthesise_sentence(sentence, turn_id, index))
        self._tts_tasks.add(task)
        task.add_done_callback(self._tts_tasks.discard)

    async def _synthesise_sentence(self, sentence: str, turn_id: str, index: int) -> None:
        async with self._tts_order_lock:
            if turn_id in self._cancelled_turns:
                return
            await self.send_state("speaking", turn_id=turn_id)
            with tempfile.TemporaryDirectory(prefix="hermes-voice-tts-") as tmp:
                wav_in = str(Path(tmp) / "sentence-tts.wav")
                wav_out = str(Path(tmp) / "sentence-pcm.wav")
                from tools.tts_tool import text_to_speech_tool
                raw = await asyncio.to_thread(text_to_speech_tool, sentence, wav_in)
                result = json.loads(raw)
                if not result.get("success"):
                    raise RuntimeError(result.get("error") or "TTS failed")
                source = Path(result.get("file_path") or wav_in)
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-loglevel", "error", "-y", "-i", str(source),
                    "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                    wav_out,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _out, err = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(f"PCM conversion failed: {err.decode(errors='replace')[:200]}")
                pcm = Path(wav_out).read_bytes()
            if turn_id in self._cancelled_turns:
                return
            await self.send_json(
                "tts_audio_meta", turn_id=turn_id, sentence_index=index,
                codec="pcm_s16le", container="raw", text=sentence,
                sample_rate=16000, channels=1,
            )
            async with self._send_lock:
                await self.ws.send_bytes(pcm)

    @staticmethod
    def _transcribe_pcm(pcm: bytes) -> str:
        with tempfile.TemporaryDirectory(prefix="hermes-voice-asr-") as tmp:
            wav_path = Path(tmp) / "utterance.wav"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(VOICE_CHANNELS)
                wav.setsampwidth(VOICE_SAMPLE_WIDTH)
                wav.setframerate(VOICE_SAMPLE_RATE)
                wav.writeframes(pcm)
            try:
                from tools.transcription_tools import transcribe_audio
                result = transcribe_audio(str(wav_path))
            except Exception as exc:
                logger.exception("voice STT tool raised: %s", exc)
                raise
            if not result.get("success"):
                logger.warning(
                    "voice STT returned no_speech file=%s error=%s wav_bytes=%d",
                    wav_path, result.get("error"), len(pcm),
                )
                raise RuntimeError(result.get("error") or "STT failed")
            text = str(result.get("transcript") or "").strip()
            logger.info("voice STT result chars=%d preview=%r", len(text), text[:80])
            return text


async def handle_voice_websocket(request: Any, adapter: Any) -> Any:
    """aiohttp route handler for GET /v1/voice/ws."""
    from aiohttp import WSMsgType, web

    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    session_id = request.headers.get("X-Hermes-Session-Id", "").strip() or f"voice_{uuid.uuid4().hex}"
    session_key, key_err = adapter._parse_session_key_header(request)
    if key_err is not None:
        return key_err

    ws = web.WebSocketResponse(heartbeat=20.0, max_msg_size=MAX_UTTERANCE_BYTES + 1024)
    await ws.prepare(request)
    session = VoiceSession(ws=ws, adapter=adapter, session_id=session_id, session_key=session_key)
    await session.send_json(
        "session_ready", codec="pcm_s16le", sample_rate=VOICE_SAMPLE_RATE,
        channels=VOICE_CHANNELS, tts_codec="opus", tts_container="ogg",
    )
    await session.send_state("listening")

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                try:
                    await session.accept_audio(msg.data)
                except VoiceProtocolError as exc:
                    await session.send_json("error", code="invalid_audio", message=str(exc))
            elif msg.type == WSMsgType.TEXT:
                try:
                    frame = json.loads(msg.data)
                    frame_type = frame.get("type")
                    if frame_type == "commit_audio":
                        await session.commit_audio()
                    elif frame_type == "barge_in":
                        await session.barge_in()
                    elif frame_type == "ping":
                        await session.send_json("pong", timestamp=time.time())
                    else:
                        raise VoiceProtocolError(f"unknown frame type: {frame_type}")
                except (json.JSONDecodeError, VoiceProtocolError) as exc:
                    await session.send_json("error", code="invalid_frame", message=str(exc))
            elif msg.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                break
    finally:
        await session.close()
    return ws
