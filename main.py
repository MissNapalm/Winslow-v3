#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming Claude chatbot + local Whisper mic transcription (faster-whisper) + image analysis

Modes:
  • Typing (default):    python3 main.py
  • Voice mode (ASR):    python3 main.py --voice

What it does:
  - Records mic audio until you press Enter (no silence/VAD auto-stop)
  - Transcribes locally using faster-whisper
  - Streams Claude response and speaks sentences as they arrive
  - Loads character prompt from prompt.txt
  - Analyzes images from /images/ directory when referenced
  - Typewriter output, strips *stage directions*
  - ⎵ Spacebar interrupt: stops streaming + cuts current speech immediately

ENV (.env next to this file):
  CLAUDE_API_KEY=sk-ant-...
  MODEL=claude-3-5-sonnet-20241022   # optional override
  TEMPERATURE=0.7                    # optional
  MAX_TOKENS=800                     # optional
  WHISPER_MODEL=base                 # tiny/base/small/medium/large-v3 (optional)
  WHISPER_DEVICE=auto                # auto/cuda/cpu (optional)
  WHISPER_COMPUTE=float16            # float16/int8 (optional)
"""

import os
import sys
import re
import time
import math
import queue
import argparse
import threading
import subprocess
import platform
import base64
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv, find_dotenv
import anthropic

# silence SDK/model deprecation chatter
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------- Colors and Visual Effects ----------
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    BG_BLACK = '\033[40m'
    BG_GRAY = '\033[100m'

def typewriter_print(text: str, color: str = "", delay: float = 0.02, end: str = ""):
    """Print text with a typewriter effect (keeps whitespace intact)."""
    if color:
        sys.stdout.write(color)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        if char not in (' ', '\n', '\t'):
            time.sleep(delay)
    if color:
        sys.stdout.write(Colors.RESET)
    if end:
        sys.stdout.write(end)
    sys.stdout.flush()

def print_header():
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print("🤖 CLAUDE VOICE ASSISTANT")
    print(f"{'='*60}{Colors.RESET}\n")

def print_status(message: str, status_type: str = "info"):
    colors = {"info": Colors.BLUE, "success": Colors.GREEN, "warning": Colors.YELLOW, "error": Colors.RED}
    color = colors.get(status_type, Colors.BLUE)
    print(f"{color}● {message}{Colors.RESET}")

def print_user_input(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}You:{Colors.RESET} {text}")

def print_assistant_header():
    print(f"\n{Colors.YELLOW}{Colors.BOLD}Assistant:{Colors.RESET}", end=" ")

# ---------- Cleaners (display vs TTS) ----------
def clean_text_for_display(text: str) -> str:
    """
    Display cleaner: remove *stage directions* but DO NOT collapse or strip whitespace.
    This preserves spaces between streamed chunks.
    """
    if text is None:
        return ""
    return re.sub(r'\*[^*]*\*', '', text)

def clean_text_for_tts(text: str) -> str:
    """
    TTS cleaner: remove *stage directions* and normalize whitespace so it sounds natural.
    Safe to trim here because we feed whole sentences.
    """
    if text is None:
        return ""
    text = re.sub(r'\*[^*]*\*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- .env + config ----------
dotenv_path = find_dotenv(".env", usecwd=True)
load_dotenv(dotenv_path, override=False)

API_KEY = os.getenv("CLAUDE_API_KEY", "")
MODEL = os.getenv("MODEL", "claude-3-5-sonnet-20241022")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7") or 0.7)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800") or 800)

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto|cuda|cpu
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")  # float16|int8

print_header()
print_status(f"Config loaded from: {dotenv_path or 'environment'}")
print_status(f"Claude API Key: {'✓ Present' if API_KEY else '✗ Missing'}", "success" if API_KEY else "error")
if not API_KEY:
    print_status("Add CLAUDE_API_KEY=sk-ant-... to your .env file", "error")
    sys.exit(1)

# ---------- Image utilities ----------
def load_system_prompt() -> str:
    prompt_file = Path("prompt.txt")
    if prompt_file.exists():
        try:
            text = prompt_file.read_text(encoding='utf-8').strip()
            print_status(f"Character prompt loaded from {prompt_file} ({len(text)} chars)", "success")
            return text
        except Exception as e:
            print_status(f"Error reading prompt.txt: {e}", "error")
    print_status("Using default system prompt (no prompt.txt found)", "warning")
    return (
        "You are a concise, friendly assistant. "
        "Write clearly. If you list steps, keep them short."
    )

def get_image_files() -> List[Path]:
    images_dir = Path("images")
    if not images_dir.exists():
        return []
    exts = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]
    files: List[Path] = []
    for ext in exts:
        files.extend(images_dir.glob(ext))
        files.extend(images_dir.glob(ext.upper()))
    return sorted(files)

def encode_image_base64(image_path: Path) -> Optional[str]:
    try:
        data = image_path.read_bytes()
        import base64 as _b64
        return _b64.b64encode(data).decode("utf-8")
    except Exception as e:
        print_status(f"Error encoding {image_path}: {e}", "error")
        return None

def get_media_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    return {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
    }.get(ext, 'image/jpeg')

def find_referenced_images(text: str) -> List[Path]:
    image_files = get_image_files()
    if not image_files:
        return []
    referenced: List[Path] = []
    t = text.lower()
    for p in image_files:
        if p.name.lower() in t:
            referenced.append(p)
    if any(k in t for k in ('image', 'photo', 'picture', 'img', 'pic')):
        if len(image_files) <= 3:
            for p in image_files:
                if p not in referenced:
                    referenced.append(p)
        elif not referenced:
            referenced.append(image_files[0])
    return referenced

# ---------- Spacebar watcher ----------
# ---------- Spacebar watcher (robust via /dev/tty) ----------
class SpacebarWatcher:
    """Non-blocking listener that sets .pressed when spacebar is hit (POSIX: /dev/tty; Windows: msvcrt)."""
    def __init__(self):
        self.pressed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            if platform.system() == "Windows":
                import msvcrt
                while not self._stop.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b' ',):
                            self.pressed = True
                            return
                    time.sleep(0.03)
                return

            # POSIX: read directly from controlling terminal
            import termios, tty, select, os
            try:
                fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            except Exception:
                # Fallback to stdin if /dev/tty not available
                fd = sys.stdin.fileno()

            # Save term settings (best effort)
            old = None
            try:
                old = termios.tcgetattr(fd)
            except Exception:
                pass

            try:
                # put terminal in cbreak mode (character-by-character)
                try:
                    tty.setcbreak(fd)
                except Exception:
                    pass

                while not self._stop.is_set():
                    r, _, _ = select.select([fd], [], [], 0.03)
                    if r:
                        try:
                            ch = os.read(fd, 1)
                            if ch == b' ':
                                self.pressed = True
                                return
                        except Exception:
                            pass
            finally:
                # restore settings
                if old is not None:
                    try:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    except Exception:
                        pass
                try:
                    if fd not in (0, 1, 2):
                        os.close(fd)
                except Exception:
                    pass
        except Exception:
            # quietly disable if environment doesn't allow key capture
            return

    def stop(self):
        self._stop.set()
        try:
            self._thread.join(timeout=0.2)
        except Exception:
            pass


# ---------- TTS ----------
def which(cmd: str) -> bool:
    from shutil import which as _w
    return _w(cmd) is not None

class TTS:
    """Minimal TTS queue (macOS 'say' / Linux 'espeak' / fallback prints) with interrupt."""
    def __init__(self):
        self.available = False
        self.backend = None
        self.is_speaking = False
        self._lock = threading.Lock()
        self._kill = threading.Event()
        self._current_proc: Optional[subprocess.Popen] = None

        if platform.system() == "Darwin" and which("say"):
            self.available = True
            self.backend = ("say",)
        elif platform.system() == "Linux" and which("espeak"):
            self.available = True
            self.backend = ("espeak", "-s", "170", "-p", "40")

        self.q: "queue.Queue[str]" = queue.Queue()
        self.stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop.is_set():
            try:
                item = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            self._speak_now(item)
            self.q.task_done()

    def _speak_now(self, text: str):
        text = clean_text_for_tts(text)
        if not text:
            return
        with self._lock:
            self.is_speaking = True
        try:
            if not self.available:
                print(f"\n{Colors.GRAY}🔊 {text}{Colors.RESET}\n", flush=True)
                # simulate duration so space interrupt timing feels natural
                for _ in range(int(max(1, len(text) * 0.02 / 0.05))):
                    if self._kill.is_set() or self.stop.is_set():
                        break
                    time.sleep(0.05)
                return
            # start process
            if self.backend[0] == "say":
                proc = subprocess.Popen(["say", "-r", "180", text])
            else:
                proc = subprocess.Popen([*self.backend, text])
            self._current_proc = proc
            # poll with ability to interrupt
            while proc.poll() is None and not self.stop.is_set() and not self._kill.is_set():
                time.sleep(0.05)
            if proc.poll() is None:
                # still running but we were told to stop
                try:
                    proc.terminate()
                except Exception:
                    pass
        except Exception as e:
            print_status(f"TTS error: {e}", "error")
        finally:
            self._current_proc = None
            self._kill.clear()
            with self._lock:
                self.is_speaking = False

    def say(self, text: str):
        self.q.put(text)

    def stop_now(self):
        """Immediately stop current speech and clear queued items."""
        self._kill.set()
        with self._lock:
            if self._current_proc is not None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
        # drain the queue
        try:
            while True:
                self.q.get_nowait()
                self.q.task_done()
        except queue.Empty:
            pass

    def is_currently_speaking(self) -> bool:
        with self._lock:
            return self.is_speaking

    def shutdown(self):
        self.stop.set()
        self._kill.set()
        self.q.put(None)

# ---------- Voice (mic capture + faster-whisper) ----------
class VoiceASR:
    """Local mic recording (Enter to stop) + faster-whisper transcription."""
    def __init__(self, tts_ref: Optional[TTS] = None):
        import pyaudio
        from faster_whisper import WhisperModel
        self.pyaudio = pyaudio
        self.WhisperModel = WhisperModel
        self.tts_ref = tts_ref

        # audio params
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 4096

        # safety only (no VAD)
        self.max_recording_sec = 300.0

        # load whisper
        device = self._pick_device()
        compute = WHISPER_COMPUTE
        model_name = WHISPER_MODEL_NAME
        print_status(f"Loading Whisper model: {model_name} on {device} ({compute})")
        try:
            self.model = self.WhisperModel(model_name, device=device, compute_type=compute)
            print_status("Whisper model loaded successfully", "success")
        except Exception:
            print_status("Falling back to CPU int8 for Whisper", "warning")
            self.model = self.WhisperModel(model_name, device="cpu", compute_type="int8")

    def _pick_device(self) -> str:
        if WHISPER_DEVICE.lower() in ("cuda", "cpu"):
            return WHISPER_DEVICE.lower()
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def record_once(self) -> Optional[bytes]:
        """
        Record until user presses Enter (no silence detection).
        Won't start while TTS is speaking.
        """
        if self.tts_ref:
            while self.tts_ref.is_currently_speaking():
                time.sleep(0.05)

        p = self.pyaudio.PyAudio()
        try:
            stream = p.open(format=self.format, channels=self.channels, rate=self.rate,
                            input=True, frames_per_buffer=self.chunk)
        except Exception as e:
            print_status(f"Could not open microphone: {e}", "error")
            return None

        print(f"\n{Colors.GREEN}🎤 Listening... (press Enter to stop){Colors.RESET}")
        frames: List[bytes] = []
        start = time.time()
        stop = False

        def _enter_waiter():
            nonlocal stop
            try:
                import sys, termios, tty, select
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while not stop:
                        r, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if r:
                            ch = sys.stdin.read(1)
                            if ch in ('\n', '\r', ' '):  # Enter or Space
                                stop = True
                                break
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                # fallback: blocking input
                try:
                    input()
                    stop = True
                except Exception:
                    pass

        threading.Thread(target=_enter_waiter, daemon=True).start()

        try:
            while True:
                if self.tts_ref and self.tts_ref.is_currently_speaking():
                    print_status("Stopping recording - TTS is speaking", "warning")
                    break
                if stop:
                    print_status("Stopped by user", "info")
                    break
                pcm = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(pcm)
                if (time.time() - start) > self.max_recording_sec:
                    print_status("Maximum recording time reached", "warning")
                    break
        finally:
            try:
                stream.stop_stream(); stream.close()
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass

        if not frames:
            print_status("No audio captured", "error")
            return None
        audio = b"".join(frames)
        secs = len(audio) / (self.rate * 2)
        print_status(f"Captured {secs:.1f}s of audio", "success")
        return audio

    def transcribe(self, audio: bytes) -> Optional[str]:
        if not audio or len(audio) < 2000:
            return None
        import numpy as np
        arr = np.frombuffer(audio, dtype=np.int16).astype("float32") / 32768.0
        if float(np.sqrt((arr**2).mean())) < 0.01:
            return None

        print_status("Transcribing audio...", "info")
        try:
            segments, _info = self.model.transcribe(
                arr,
                language="en",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, max_speech_duration_s=30),
                word_timestamps=False,
                initial_prompt="Casual English conversation."
            )
            parts = []
            for s in segments:
                if (s.end - s.start) < 0.25:
                    continue
                t = (s.text or "")
                if t:
                    parts.append(t.strip())
            text = " ".join(parts).strip()
            return text or None
        except Exception as e:
            print_status(f"Whisper transcription error: {e}", "error")
            return None

# ---------- Bot (stream + speak) ----------
class Bot:
    def __init__(self):
        print_status(f"Initializing Claude client (anthropic v{anthropic.__version__})", "info")
        self.client = anthropic.Anthropic(api_key=API_KEY)
        self.tts = TTS()
        self.history: List[Dict[str, Any]] = []
        self.system_prompt = load_system_prompt()
        self.sent_re = re.compile(r"([^.?!]*[.?!])")

        imgs = get_image_files()
        if imgs:
            print_status(f"Found {len(imgs)} images in /images/", "success")
            for img in imgs[:5]:
                print(f"   {Colors.GRAY}• {img.name}{Colors.RESET}")
            if len(imgs) > 5:
                print(f"   {Colors.GRAY}... and {len(imgs) - 5} more{Colors.RESET}")
        else:
            print_status("No images found in /images/ directory", "warning")

        # quick probe (kept quiet; errors show clearly)
        try:
            _ = self.client.messages.create(
                model=MODEL, max_tokens=5,
                messages=[{"role":"user","content":"OK"}]
            )
        except Exception as e:
            print_status(f"Claude connection failed: {e}", "error")
            sys.exit(1)

    def _enqueue_complete_sentences(self, buf: str) -> str:
        last_end = 0
        for m in self.sent_re.finditer(buf):
            sent = m.group(1)
            if sent:
                clean_sent = clean_text_for_tts(sent)
                if clean_sent:
                    self.tts.say(clean_sent)
            last_end = m.end()
        return buf[last_end:] if last_end else buf

    def _prepare_message_content(self, user_text: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img_path in find_referenced_images(user_text):
            b64 = encode_image_base64(img_path)
            if b64:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": get_media_type(img_path),
                        "data": b64
                    }
                })
        return content

    def chat_once(self, user_text: str):
        user_content = self._prepare_message_content(user_text)

        # store plain text in history (keep last ~12)
        self.history.append({"role": "user", "content": user_text})
        msgs: List[Dict[str, Any]] = []
        recent = self.history[-12:]
        for i, m in enumerate(recent):
            if i == len(recent) - 1:
                msgs.append({"role": m["role"], "content": user_content})
            else:
                msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})

        print_assistant_header()

        tts_buf = ""
        full_chunks: List[str] = []
        got_text = False
        interrupted = False

        watcher = SpacebarWatcher()
        try:
            with self.client.messages.stream(
                model=MODEL,
                system=self.system_prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=msgs,
            ) as stream:
                for chunk in stream.text_stream:
                    if watcher.pressed:
                        interrupted = True
                        break
                    if not chunk:
                        continue
                    got_text = True

                    # display: strip *stage directions* but DO NOT collapse whitespace
                    display_chunk = clean_text_for_display(chunk)
                    if display_chunk:
                        typewriter_print(display_chunk, Colors.YELLOW, delay=0.01)

                    # speak sentences when complete
                    tts_buf += chunk
                    tts_buf = self._enqueue_complete_sentences(tts_buf)

                    full_chunks.append(chunk)

                # if we broke due to spacebar, explicitly close stream
                stream.close()
        except Exception as e:
            print_status(f"Streaming error: {type(e).__name__}: {e}", "error")
            watcher.stop()
            return
        finally:
            watcher.stop()

        if interrupted:
            # stop any ongoing speech and drop buffered/queued lines
            self.tts.stop_now()
            print_status("⎵ Interrupted", "warning")
            # do not append leftover partial sentence or finalize full text
            print()  # newline after the partial output
            return

        if not got_text:
            print_status("Stream yielded no text, trying non-stream fallback", "warning")
            try:
                r = self.client.messages.create(
                    model=MODEL, system=self.system_prompt,
                    max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                    messages=msgs,
                )
                txt = r.content[0].text
                typewriter_print(clean_text_for_display(txt), Colors.YELLOW, delay=0.01)
                # TTS on sentence boundaries
                for s in self.sent_re.findall(txt):
                    cs = clean_text_for_tts(s)
                    if cs:
                        self.tts.say(cs)
                rem = self.sent_re.sub("", txt)
                rem_cs = clean_text_for_tts(rem)
                if rem_cs:
                    self.tts.say(rem_cs)
                self.history.append({"role": "assistant", "content": txt})
                return
            except Exception as e:
                print_status(f"Non-stream fallback error: {type(e).__name__}: {e}", "error")
                return

        # speak leftover partial sentence
        rem = tts_buf
        rem_cs = clean_text_for_tts(rem)
        if rem_cs:
            self.tts.say(rem_cs)

        full = "".join(full_chunks)
        self.history.append({"role": "assistant", "content": full})
        print()  # newline after response

    def close(self):
        self.tts.shutdown()

# ---------- CLI ----------
def parse_args():
    import argparse as _argparse
    p = _argparse.ArgumentParser(description="Enhanced Claude Chatbot with Voice & Vision")
    p.add_argument("--say", help="Send one message non-interactively and exit")
    p.add_argument("--voice", action="store_true", help="Voice mode: record + transcribe locally, then chat")
    p.add_argument("--list-images", action="store_true", help="List available images and exit")
    return p.parse_args()

def main():
    args = parse_args()

    if args.list_images:
        files = get_image_files()
        if files:
            print_status(f"Found {len(files)} images in /images/:", "success")
            for img in files:
                print(f"  {Colors.CYAN}{img.name}{Colors.RESET}")
        else:
            print_status("No images found in /images/ directory", "warning")
        return

    bot = Bot()

    if args.say:
        print_user_input(args.say)
        bot.chat_once(args.say)
        time.sleep(0.2)
        bot.close()
        return

    if args.voice:
        try:
            asr = VoiceASR(tts_ref=bot.tts)
        except Exception as e:
            print_status(f"Voice initialization failed: {e}", "error")
            bot.close()
            return

        print_status("Voice mode activated", "success")
        print_status("Mention images by name or say 'image', 'photo', 'picture' to include them", "info")
        print_status("Press Enter to stop recording; Ctrl+C to quit", "info")
        print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}")

        try:
            while True:
                audio = asr.record_once()
                if not audio:
                    continue
                text = asr.transcribe(audio)
                if not text:
                    print_status("Could not transcribe audio. Please try again.", "warning")
                    continue
                print_user_input(text)
                bot.chat_once(text)
                print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}")
        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Goodbye! 👋{Colors.RESET}")
        finally:
            bot.close()
        return

    # Interactive typing
    print_status("Interactive mode - Type your messages below", "success")
    print_status("Reference images by filename or use 'image', 'photo', 'picture'", "info")
    print_status("Use --voice for microphone input, Ctrl+C to quit", "info")
    print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}")

    try:
        while True:
            try:
                user = input(f"\n{Colors.CYAN}{Colors.BOLD}You: {Colors.RESET}").strip()
                if not user:
                    continue
                bot.chat_once(user)
                print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}")
            except KeyboardInterrupt:
                print(f"\n\n{Colors.CYAN}Goodbye! 👋{Colors.RESET}")
                break
    finally:
        bot.close()

if __name__ == "__main__":
    main()
