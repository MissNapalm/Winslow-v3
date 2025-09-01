import os
import sys
import threading
import pyaudio
import anthropic
from dotenv import load_dotenv
from io import BytesIO
import numpy as np
from faster_whisper import WhisperModel
import time
import concurrent.futures
import warnings
import subprocess
import platform
import shutil
import json
import re
import wave
import tempfile

# NEW imports for folder image analysis
from pathlib import Path
from PIL import Image
import base64
import mimetypes

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# ---------- Typewriter printer (yellow) ----------
def typewriter_print(text, delay=0.02, color="\033[93m", stop_event=None):
    """Print text with a typewriter effect in yellow by default (ANSI). Supports interruption via stop_event."""
    try:
        sys.stdout.write(color)
        sys.stdout.flush()
    except Exception:
        pass
    for ch in text:
        if stop_event and stop_event.is_set():
            break
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    # reset color + newline
    try:
        sys.stdout.write("\033[0m\n")
    except Exception:
        sys.stdout.write("\n")
    sys.stdout.flush()

class UltraFastTranscriber:
    def __init__(self):
        # Initialize Claude client
        claude_api_key = os.getenv('CLAUDE_API_KEY')
        if not claude_api_key:
            raise ValueError("CLAUDE_API_KEY not found in environment variables")
        self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
        
        # Initialize local Whisper model (GPU if available)
        print("🚀 Loading local Whisper model...")
        try:
            self.whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
            print("✅ Using GPU acceleration")
        except Exception:
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            print("✅ Using CPU (install CUDA for GPU speedup)")
        
        # Initialize TTS
        self.setup_tts()
        
        # Load base system prompt from file
        self.base_system_prompt = self.load_prompt_from_file()

        # Conversation memory (persisted)
        self.history = []             # list of {"role": "user"|"assistant", "content": str}
        self.running_summary = ""     # rolling summary of older context
        self.max_history_chars = 14000
        self.memory_path = "conversation.json"
        self._load_memory()

        # NEW: images folder (for vision commands)
        self.images_dir = Path(os.getenv("IMAGES_DIR", "images")).resolve()
        self.images_dir.mkdir(parents=True, exist_ok=True)
        print(f"🖼️  Images directory: {self.images_dir}")
        
        # IMPROVED audio settings for better quality
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Standard rate for Whisper
        self.chunk = 4096  # Smaller chunks for better responsiveness
        self.recording = False
        self.min_recording_duration = 0.5  # Minimum 0.5 seconds
        self.silence_threshold = 500  # RMS threshold for silence detection
        self.silence_duration = 1.5  # Stop after 1.5s of silence
        
        # Threading executor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

        # Keyboard raw mode support (for spacebar)
        self._raw_supported = self._detect_raw_mode_support()

        # Answer interruption support
        self.answer_interrupt_event = threading.Event()

    # ---------- Memory management ----------
    def _cleanup_memory(self):
        """Remove any empty or invalid messages from history."""
        original_count = len(self.history)
        self.history = [
            m for m in self.history 
            if (isinstance(m, dict) and 
                m.get("role") in ("user", "assistant") and 
                m.get("content") and 
                str(m.get("content")).strip())
        ]
        cleaned_count = len(self.history)
        
        if original_count != cleaned_count:
            print(f"🧹 Cleaned {original_count - cleaned_count} empty messages from memory")
            self._save_memory()

    def _load_memory(self):
        try:
            if os.path.exists(self.memory_path):
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.history = data.get("history", [])
                self.running_summary = data.get("running_summary", "")
                # Clean up any empty messages from loaded data
                self._cleanup_memory()
                print("🧠 Loaded conversation memory.")
            
            # Try to load memories.json for additional context about past interactions
            memories_path = "memories.json"
            if os.path.exists(memories_path):
                try:
                    with open(memories_path, "r", encoding="utf-8") as f:
                        memories_data = json.load(f)
                    
                    # Extract facts from memories file
                    facts = memories_data.get("facts", [])
                    if facts:
                        fact_summary = "\n\nUser facts from previous conversations:\n" + "\n".join(f"- {fact}" for fact in facts)
                        self.running_summary += fact_summary
                        print(f"🧠 Loaded {len(facts)} facts from memories.json")
                    
                    # If history is empty, import some past messages
                    if not self.history:
                        past_messages = memories_data.get("messages", [])
                        if past_messages:
                            # Import up to 10 recent message pairs (20 messages total)
                            imported_count = min(20, len(past_messages))
                            recent_messages = past_messages[-imported_count:]
                            
                            # Add these to history
                            self.history.extend(recent_messages)
                            print(f"🧠 Imported {imported_count} messages from memories.json")
                except Exception as e:
                    print(f"⚠️ Error loading memories.json: {e}")
                    
        except Exception as e:
            print(f"⚠️ Could not load memory: {e}")
            self.history = []
            self.running_summary = ""

    def _save_memory(self):
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"history": self.history, "running_summary": self.running_summary},
                    f, ensure_ascii=False, indent=2
                )
        except Exception as e:
            print(f"⚠️ Could not save memory: {e}")

    def _maybe_summarize_history(self):
        """When history gets big, summarize the oldest half into running_summary."""
        try:
            # Filter out any empty messages before checking size
            self.history = [
                m for m in self.history 
                if (isinstance(m, dict) and 
                    m.get("role") in ("user", "assistant") and 
                    m.get("content") and 
                    str(m.get("content")).strip())
            ]
            
            serialized = json.dumps(self.history, ensure_ascii=False)
            if len(serialized) <= self.max_history_chars or len(self.history) < 6:
                return

            cut = max(3, len(self.history) // 2)
            old_chunk = self.history[:cut]
            self.history = self.history[cut:]

            prompt = (
                "Summarize the prior conversation turns below into 8–12 concise bullet points. "
                "Capture user preferences, facts about the user, ongoing tasks, decisions, and unresolved threads. "
                "Keep it neutral and compact.\n\n" + json.dumps(old_chunk, ensure_ascii=False)
            )
            resp = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                temperature=0.2,
                system="You are a precise summarizer.",
                messages=[{"role": "user", "content": prompt}]
            )
            summary = resp.content[0].text.strip()
            
            # Only add summary if it's non-empty
            if summary:
                self.running_summary += (("\n" if self.running_summary else "") + summary)
                print("🧠 Summarized older history.")
            else:
                print("⚠️ Summary was empty, skipping...")
                
        except Exception as e:
            print(f"⚠️ Could not summarize: {e}")

    def _reset_memory(self):
        self.history = []
        self.running_summary = ""
        self._save_memory()
        print("🧼 Memory reset.")

    # ---------- TTS ----------
    def setup_tts(self):
        self.tts_available = False
        if platform.system() == 'Darwin':
            if shutil.which('say'):
                self.tts_available = True
                print("✅ macOS system TTS available")
        elif platform.system() == 'Linux':
            if shutil.which('espeak'):
                self.tts_available = True
                print("✅ Linux espeak TTS available")

    def clean_text_for_tts(self, text):
        """Remove visual elements that don't speak well but keep the text readable."""
        if not text:
            return ""
        
        # Remove common emoticons and visual elements
        tts_text = text
        tts_text = re.sub(r'>:\]', '', tts_text)
        tts_text = re.sub(r':\]', '', tts_text) 
        tts_text = re.sub(r'\[.*?\]', '', tts_text)
        tts_text = re.sub(r'<.*?>', '', tts_text)
        tts_text = re.sub(r'[>]{2,}', '', tts_text)  # >>
        tts_text = re.sub(r'[*]{1,2}[^*]*[*]{1,2}', '', tts_text)  # *action*
        tts_text = ' '.join(tts_text.split()).strip()
        return tts_text

    def speak_system(self, text):
        if not self.tts_available:
            # Still return True to avoid blocking workflows that wait for TTS
            print(f"🔊 Character would say: {text}")
            return True
        
        # Clean text for TTS while keeping original for display
        tts_text = self.clean_text_for_tts(text)
        
        try:
            proc = None
            if platform.system() == 'Darwin':
                proc = subprocess.Popen(['say', '-v', 'Jamie (Enhanced)', '-r', '180', tts_text])
            elif platform.system() == 'Linux':
                proc = subprocess.Popen(['espeak', '-s', '160', '-p', '40', tts_text])
            if proc:
                while proc.poll() is None:
                    if self.answer_interrupt_event.is_set():
                        proc.terminate()
                        break
                    time.sleep(0.05)
            return True
        except Exception as e:
            print(f"❌ System TTS error: {e}")
            print(f"🔊 Character would say: {tts_text}")
            return False

    def speak_async(self, text):
        """Start OS TTS in the background so it overlaps with the typewriter. Respects interruption."""
        t = threading.Thread(target=self.speak_system, args=(text,), daemon=True)
        t.start()
        return t

    # ---------- Prompt ----------
    def load_prompt_from_file(self, prompt_file_path="prompt.txt"):
        try:
            if os.path.exists(prompt_file_path):
                with open(prompt_file_path, 'r', encoding='utf-8') as file:
                    prompt = file.read().strip()
                    if prompt:
                        print(f"✅ Loaded character prompt from {prompt_file_path}")
                        
                        # Check if memories.json exists to augment the prompt
                        memories_path = "memories.json"
                        if os.path.exists(memories_path):
                            try:
                                with open(memories_path, 'r', encoding='utf-8') as f:
                                    memories_data = json.load(f)
                                
                                # Extract facts from memories.json
                                facts = memories_data.get("facts", [])
                                if facts:
                                    facts_str = "\n\nImportant facts about our past interactions:\n" + "\n".join(f"- {fact}" for fact in facts[:5])
                                    
                                    # Add memory context to prompt
                                    prompt += facts_str
                                    print(f"✅ Added {len(facts[:5])} memory facts to prompt")
                            except Exception as e:
                                print(f"⚠️ Error processing memories.json: {e}")
                        
                        return prompt
            return "You are a helpful AI assistant. Keep responses concise and conversational."
        except Exception as e:
            print(f"❌ Error reading prompt file: {e}")
            return "You are a helpful AI assistant. Keep responses concise and conversational."

    # ---------- IMPROVED Audio processing ----------
    def calculate_rms(self, data):
        """Calculate RMS (root mean square) for volume detection."""
        try:
            audio_np = np.frombuffer(data, dtype=np.int16)
            return np.sqrt(np.mean(audio_np ** 2))
        except:
            return 0

    def _detect_raw_mode_support(self):
        if platform.system() == "Windows":
            return True  # we'll use msvcrt
        # POSIX: require a TTY
        return sys.stdin.isatty()

    # POSIX raw mode helpers
    def _posix_raw_reader(self, stop_flag):
        """Read single chars in raw mode on a background thread; stop on SPACE/ENTER."""
        import termios, tty, select
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # cbreak so we get chars immediately
            while self.recording and not stop_flag.is_set():
                rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch in (' ', '\n', '\r'):
                        self.stop_recording()
                        stop_flag.set()
                        break
        except Exception:
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

    def record_audio_optimized(self):
        """Enhanced recording with better silence detection and quality."""
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
        except Exception as e:
            print(f"❌ Could not open audio stream: {e}")
            return None

        # UX hint
        if self._raw_mode_is_windows():
            print("🎤 Recording... Press SPACE (or Enter) to stop, or speak and pause.")
        elif self._raw_supported:
            print("🎤 Recording... Press SPACE (or Enter) to stop, or speak and pause.")
        else:
            print("🎤 Recording... Press Enter to stop (raw keys unavailable).")

        self.recording = True
        audio_frames = []
        start_time = time.time()
        last_sound_time = time.time()
        has_sound = False

        # Start a raw key listener where possible
        stop_flag = threading.Event()
        key_thread = None

        if self._raw_mode_is_windows():
            import msvcrt
        elif self._raw_supported:
            key_thread = threading.Thread(target=self._posix_raw_reader, args=(stop_flag,), daemon=True)
            key_thread.start()
        else:
            def _enter_waiter():
                try:
                    input()
                except Exception:
                    pass
                self.stop_recording()
                stop_flag.set()
            key_thread = threading.Thread(target=_enter_waiter, daemon=True)
            key_thread.start()

        try:
            while self.recording:
                current_time = time.time()
                
                # Handle keyboard input for Windows
                if self._raw_mode_is_windows():
                    import msvcrt
                    if msvcrt.kbhit():
                        try:
                            ch = msvcrt.getch()
                            if ch in (b' ', b'\r', b'\n'):
                                self.stop_recording()
                                break
                        except Exception:
                            pass

                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    audio_frames.append(data)
                    
                    # Calculate volume for silence detection
                    rms = self.calculate_rms(data)
                    
                    if rms > self.silence_threshold:
                        has_sound = True
                        last_sound_time = current_time
                        
                    # Auto-stop conditions
                    recording_duration = current_time - start_time
                    silence_duration = current_time - last_sound_time
                    
                    # Stop if we've been silent for too long after detecting sound
                    if has_sound and silence_duration > self.silence_duration and recording_duration > self.min_recording_duration:
                        print("🔇 Auto-stopping due to silence...")
                        self.stop_recording()
                        break
                    
                    # Maximum recording time safety net
                    if recording_duration > 500:  # 500 second max
                        print("⏰ Max recording time reached...")
                        self.stop_recording()
                        break
                        
                except Exception as e:
                    print(f"⚠️ Audio read error: {e}")
                    continue
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
            try:
                audio.terminate()
            except:
                pass
            # ensure key thread cleaned up
            stop_flag.set()
            if key_thread and key_thread.is_alive():
                try:
                    key_thread.join(timeout=0.1)
                except Exception:
                    pass

        if not audio_frames:
            print("❌ No audio captured")
            return None
            
        audio_data = b''.join(audio_frames)
        duration = len(audio_data) / (self.rate * 2)  # 2 bytes per sample for int16
        print(f"📊 Captured {duration:.1f}s of audio")
        
        return audio_data

    def _raw_mode_is_windows(self):
        return platform.system() == "Windows"

    def stop_recording(self):
        self.recording = False
    
    def save_audio_debug(self, audio_data, filename="debug_audio.wav"):
        """Save audio data to WAV file for debugging."""
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(self.rate)
                wav_file.writeframes(audio_data)
            print(f"🔍 Debug audio saved to {filename}")
        except Exception as e:
            print(f"❌ Could not save debug audio: {e}")

    def transcribe_audio_local(self, audio_data):
        """Improved transcription with better preprocessing."""
        try:
            if not audio_data or len(audio_data) < 1000:  # Less than ~0.03 seconds
                print("❌ Audio too short for transcription")
                return None
            
            # Convert to float32 numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Check if audio has any meaningful content
            rms = np.sqrt(np.mean(audio_np ** 2))
            if rms < 0.01:  # Very quiet audio
                print("❌ Audio too quiet for transcription")
                return None
            
            print(f"🔍 Audio RMS: {rms:.4f}, Length: {len(audio_np)/16000:.2f}s")
            
            # For debugging: save the audio file
            # self.save_audio_debug(audio_data)
            
            # Enhanced transcription parameters
            segments, info = self.whisper_model.transcribe(
                audio_np,
                language="en",
                beam_size=5,  # Better beam search
                best_of=5,    # Try multiple candidates
                temperature=0.0,
                word_timestamps=True,  # Enable word-level timestamps
                vad_filter=True,      # Voice activity detection
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    max_speech_duration_s=30
                ),
                initial_prompt="Hello, how are you today? I'm doing well, thanks for asking. "
                             "This is a natural conversation with clear speech."
            )
            
            # Collect segments with confidence filtering
            text_segments = []
            for segment in segments:
                # Skip very short segments (likely noise)
                if segment.end - segment.start < 0.3:
                    continue
                    
                # Skip segments with very low confidence (if available)
                segment_text = segment.text.strip()
                if segment_text and len(segment_text) > 1:
                    text_segments.append(segment_text)
            
            if not text_segments:
                print("❌ No meaningful speech detected")
                return None
                
            text = " ".join(text_segments)
            
            # Enhanced cleaning
            text = self.clean_transcription(text)
            
            print(f"🎯 Transcription confidence: {info.language_probability:.2f}")
            
            return text if text.strip() else None
            
        except Exception as e:
            print(f"❌ Local transcription error: {e}")
            return None

    def clean_transcription(self, text: str) -> str:
        """Enhanced transcription cleaning."""
        if not text:
            return ""
        
        cleaned = text.strip()
        
        # Remove common Whisper hallucinations
        hallucinations = [
            r'\bthanks for watching\b',
            r'\bsubscribe\b',
            r'\blike and subscribe\b',
            r'\bwww\.',
            r'\bmusic\]',
            r'\[music\]',
            r'\[applause\]',
            r'\[laughter\]',
            r'\[inaudible\]',
            r'\[silence\]',
            r'^\s*\.',  # Leading period
            r'^\s*,',   # Leading comma
        ]
        
        for pattern in hallucinations:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove punctuation words transcribed incorrectly
        punctuation_words = [
            (r'\bcomma\b', ','),
            (r'\bperiod\b', '.'),  
            (r'\bfull stop\b', '.'),
            (r'\bquestion mark\b', '?'),
            (r'\bexclamation mark\b', '!'),
            (r'\bexclamation point\b', '!'),
            (r'\bcolon\b', ':'),
            (r'\bsemicolon\b', ';'),
            (r'\bdash\b', '-'),
            (r'\bhyphen\b', '-'),
            (r'\bquote\b', '"'),
            (r'\bunquote\b', '"'),
            (r'\bopen paren\b', '('),
            (r'\bclose paren\b', ')'),
            (r'\bopen parenthesis\b', '('),
            (r'\bclose parenthesis\b', ')'),
        ]
        
        # Apply replacements
        for word_pattern, punctuation in punctuation_words:
            cleaned = re.sub(word_pattern, punctuation, cleaned, flags=re.IGNORECASE)
        
        # Remove standalone punctuation words at the beginning
        cleaned = re.sub(r'^(comma|period|question mark|exclamation|colon|semicolon)\s+', '', cleaned, flags=re.IGNORECASE)
        
        # Remove all consecutive repeated words (strict, prevents looping)
        words = cleaned.split()
        filtered_words = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() != words[i-1].lower():
                filtered_words.append(word)
        cleaned = ' '.join(filtered_words)
        
        # Clean up extra spaces around punctuation
        cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Ensure first letter is capitalized if it's a sentence
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        # Remove if it's too short or just punctuation
        if len(cleaned.replace(' ', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '')) < 2:
            return ""
        
        return cleaned

    # ---------- Helpers ----------
    def clean_response(self, response: str) -> str:
        """Enhanced cleaning to handle common speech artifacts and remove all emoticons."""
        if not response:
            return ""
        
        cleaned = response.strip()
        
        # Remove ALL emoticons and visual elements
        cleaned = re.sub(r'>:\]', '', cleaned)
        cleaned = re.sub(r':\]', '', cleaned)
        cleaned = re.sub(r':\)', '', cleaned)
        cleaned = re.sub(r':\(', '', cleaned)
        cleaned = re.sub(r';-?\)', '', cleaned)  # ;) or ;-)
        cleaned = re.sub(r':-?\)', '', cleaned)  # :) or :-)
        cleaned = re.sub(r':-?\(', '', cleaned)  # :( or :-(
        cleaned = re.sub(r':-?[DdPpOo]', '', cleaned)  # :D, :P, :O, etc.
        cleaned = re.sub(r'\[.*?\]', '', cleaned)  # [smirk], [smile], etc.
        cleaned = re.sub(r'<.*?>', '', cleaned)   # <grin>, <wink>, etc.
        
        # Remove action asterisks like *smiles*
        cleaned = re.sub(r'\*[^*]*\*', '', cleaned)
        
        # Remove other visual markers
        cleaned = re.sub(r'[>]{2,}', '', cleaned)  # Remove >> markers
        cleaned = re.sub(r'[~]{2,}', '', cleaned)  # Remove ~~ markers
        
        # Remove common speech artifacts and unwanted prefixes
        prefixes_to_remove = [
            r'^,\s*',           # Remove leading comma
            r'^comma\s*',       # Remove "comma" at start
            r'^um\s*',          # Remove "um"
            r'^uh\s*',          # Remove "uh"
            r'^well\s*',        # Remove "well"
            r'^so\s*',          # Remove "so" at start
            r'^okay\s*',        # Remove "okay" at start (when inappropriate)
        ]
        
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and punctuation spacing
        cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)  # Fix spacing before punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()      # Normalize all whitespace
        
        # Ensure first letter is capitalized if it's a sentence
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned

    def _build_system_prompt(self) -> str:
        """Combine base system prompt + running memory into a single top-level system string."""
        base_prompt = self.base_system_prompt
        
        # Add instruction to avoid unwanted prefixes
        speech_instruction = (
            "\n\nIMPORTANT: Respond naturally and conversationally. "
            "Do NOT start responses with words like 'comma', 'um', 'uh', 'well', or other speech artifacts. "
            "Jump straight into your response content."
        )
        
        if self.running_summary:
            return (
                f"{base_prompt}{speech_instruction}\n\n"
                f"--- Memory (summarized context) ---\n"
                f"{self.running_summary}\n"
                f"--- End Memory ---"
            )
        return f"{base_prompt}{speech_instruction}"

    # ---------- NEW: Folder Image Recognition ----------
    def _prepare_image_for_api(self, image_path: Path):
        """
        Convert image to RGB JPEG (quality 85), base64-encode.
        Returns (media_type, base64_str).
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=85, optimize=True)
            data = buf.getvalue()
        b64 = base64.b64encode(data).decode("utf-8")
        return "image/jpeg", b64

    def _list_images(self):
        pats = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp"]
        files = []
        for pat in pats:
            files.extend(self.images_dir.glob(pat))
        files = [f for f in files if f.is_file()]
        return sorted(files, key=lambda p: p.stat().st_mtime)

    def analyze_image_with_claude(self, user_text: str, image_path: Path) -> str:
        """
        Send the image + prompt to Claude (vision) and return the response.
        """
        try:
            media_type, b64 = self._prepare_image_for_api(image_path)

            # Build short recent context
            conv = []
            for m in self.history[-10:]:
                if (isinstance(m, dict) and m.get("role") in ("user", "assistant")
                    and m.get("content") and str(m.get("content")).strip()):
                    conv.append({"role": m["role"], "content": [{"type":"text","text": str(m["content"]).strip()}]})

            # Append this image turn
            content_blocks = [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64}
                },
                {"type": "text", "text": user_text or "Describe this image in detail."}
            ]
            conv.append({"role": "user", "content": content_blocks})

            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=600,
                temperature=0.2,
                system=self._build_system_prompt(),
                messages=conv
            )
            raw = response.content[0].text.strip()
            cleaned = self.clean_response(raw)

            # Save to memory
            self.history.append({"role": "user", "content": f"[image:{image_path.name}] {user_text or 'Describe this image.'}"})
            if cleaned:
                self.history.append({"role": "assistant", "content": cleaned})
                self._save_memory()

            return cleaned or "I couldn’t describe that image."
        except Exception as e:
            print(f"❌ Vision error: {e}")
            return "Sorry, I couldn’t analyze that image."

    def analyze_latest_image(self, prompt: str = "Describe this image in detail.") -> str:
        imgs = self._list_images()
        if not imgs:
            return "I can’t find any images in the images folder."
        latest = imgs[-1]
        print(f"🕵️ Analyzing latest image: {latest.name}")
        return self.analyze_image_with_claude(prompt, latest)

    # ---------- UPDATED: Voice commands ----------
    def _maybe_handle_voice_command(self, text: str):
        """
        Handle local commands:
          - 'reset memory' / 'clear memory' / 'forget everything' / 'wipe memory'
          - 'analyze the image'  -> analyze latest image
          - 'analyze image <filename>' -> analyze specific file in images folder
        Returns a string response if handled, or False/None if not.
        """
        if not text:
            return False
        t = text.strip().lower()

        # Memory reset
        if any(cmd in t for cmd in ["reset memory", "clear memory", "forget everything", "wipe memory"]):
            self._reset_memory()
            print("🗑️  Memory cleared by voice command.")
            return "Okay — I've cleared our memory."

        # Analyze image commands
        if re.search(r'\banalyze (the )?image\b', t):
            # Specific filename?
            m = re.search(r'\banalyze image\s+([^\s].+)$', t)
            if m:
                name = m.group(1).strip().strip('"').strip("'")
                path = (self.images_dir / name)
                if not path.exists():
                    return f"I couldn’t find {name} in the images folder."
                return self.analyze_image_with_claude("Describe this image in detail.", path)
            # Otherwise, latest
            return self.analyze_latest_image("Describe this image in detail.")

        return False

    # ---------- Claude ----------
    def get_claude_response(self, text):
        # Check for local commands first (now includes image analysis)
        local = self._maybe_handle_voice_command(text)
        if local:
            return local

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🤖 Getting Claude response (attempt {attempt + 1}/{max_retries})...")

                # Append new user turn and maybe summarize
                self.history.append({"role": "user", "content": text})
                self._maybe_summarize_history()

                # Build messages: ONLY user/assistant roles with non-empty content
                conv = []
                for m in self.history[-20:]:
                    if (isinstance(m, dict) and 
                        m.get("role") in ("user", "assistant") and 
                        m.get("content") and 
                        str(m.get("content")).strip()):
                        conv.append({
                            "role": m["role"],
                            "content": str(m["content"]).strip()
                        })

                # Ensure we have at least one message
                if not conv:
                    conv = [{"role": "user", "content": text}]

                # Top-level system carries base prompt + running memory + speech instructions
                system_str = self._build_system_prompt()

                response = self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=200,
                    temperature=0.7,
                    system=system_str,   # ✅ top-level system with speech instructions
                    messages=conv        # ✅ only 'user'|'assistant' with non-empty content
                )
                raw = response.content[0].text.strip()
                cleaned = self.clean_response(raw)

                # Double-check for comma prefix after cleaning
                if cleaned.lower().startswith('comma'):
                    cleaned = cleaned[5:].strip()  # Remove "comma" and any following space
                    if cleaned and cleaned[0].islower():
                        cleaned = cleaned[0].upper() + cleaned[1:]

                # Save assistant reply and persist (only if non-empty)
                if cleaned.strip():
                    self.history.append({"role": "assistant", "content": cleaned})
                    self._save_memory()

                return cleaned
            except Exception as e:
                print(f"❌ Claude error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
        return "Sorry, I'm having trouble connecting right now. Can you try again?"

    def parallel_process(self, transcription):
        """Get reply, start TTS + typewriter together, and WAIT until TTS finishes before returning.\n   SPACE interrupts both; otherwise, let TTS finish naturally."""
        try:
            self.answer_interrupt_event.clear()
            reply = self.executor.submit(self.get_claude_response, transcription).result(timeout=45)
            if reply:
                # start TTS
                tts_thread = self.speak_async(reply)

                # watch for SPACE to interrupt
                stop_event = self.answer_interrupt_event
                watcher = threading.Thread(target=self._watch_spacebar_interrupt, args=(stop_event,), daemon=True)
                watcher.start()

                # typewriter prints fully unless user presses SPACE
                typewriter_print(f"🤖 Character: {reply}", delay=0.02, stop_event=stop_event)

                # ❌ Do NOT set stop_event here. That was cutting TTS short.
                # stop_event.set()  # <-- remove this line

                # let TTS finish unless it was interrupted by SPACE
                if tts_thread is not None:
                    tts_thread.join()

                # clean up watcher
                watcher.join(timeout=0.1)

                if stop_event.is_set():
                    print("⏹️  Interrupted by spacebar. Listening again...")
            return reply

        except concurrent.futures.TimeoutError:
            fallback = "Sorry, I'm taking too long to think. Can you try again?"
            tts_thread = self.speak_async(fallback)

            stop_event = self.answer_interrupt_event
            watcher = threading.Thread(target=self._watch_spacebar_interrupt, args=(stop_event,), daemon=True)
            watcher.start()

            typewriter_print(f"🤖 Character: {fallback}", delay=0.02, stop_event=stop_event)

            # ❌ Do NOT set stop_event here either.
            # stop_event.set()  # <-- remove this line

            if tts_thread is not None:
                tts_thread.join()
            watcher.join(timeout=0.1)
            return fallback

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    def _watch_spacebar_interrupt(self, stop_event):
        """Watch for spacebar to interrupt answer (typewriter/TTS)."""
        try:
            if platform.system() == "Windows":
                import msvcrt
                while not stop_event.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b' ',):
                            stop_event.set()
                            break
                    time.sleep(0.05)
            else:
                import sys, select, termios, tty
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while not stop_event.is_set():
                        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                            ch = sys.stdin.read(1)
                            if ch == ' ':
                                stop_event.set()
                                break
                        time.sleep(0.05)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass

    # ---------- Workflow ----------
    def ultra_fast_workflow(self):
        # Start recording directly; stop via SPACE or Enter (raw), or Enter fallback
        self.audio_buffer = self.record_audio_optimized()

        if not self.audio_buffer:
            return None, None

        raw_transcription = self.transcribe_audio_local(self.audio_buffer)
        if raw_transcription:
            print(f"📝 You said: {raw_transcription}")
            # parallel_process waits for TTS to complete before returning
            response = self.parallel_process(raw_transcription)
            return raw_transcription, response
        else:
            print("❌ Could not transcribe audio - please try again")
            return None, None

def main():
    print("⚡ ULTRA-FAST Voice Character System")
    print("=" * 40)
    try:
        transcriber = UltraFastTranscriber()
        print("✅ Ultra-fast system ready!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    if transcriber.tts_available:
        print("🔊 System TTS enabled")
    else:
        print("⚠️  No TTS available - text output only")
        if platform.system() == 'Linux':
            print("   • sudo apt-get install espeak")
        elif platform.system() not in ['Darwin', 'Linux']:
            print("   • Use macOS or Linux for system TTS")

    print("\n🎯 Ready! Speak — press SPACE (or Enter) to stop, or just pause after speaking.")
    print("   Try: “analyze the image” or “analyze image myphoto.jpg” (from ./images)")
    try:
        while True:
            try:
                transcription, response = transcriber.ultra_fast_workflow()
                if transcription and response:
                    print("=" * 60)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in workflow: {e}")
                continue
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        try:
            transcriber.executor.shutdown(wait=False)
        except Exception:
            pass

if __name__ == "__main__":
    main()
