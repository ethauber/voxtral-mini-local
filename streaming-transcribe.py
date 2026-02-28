"""Continuous streaming speech-to-text with VAD-based phrase detection."""

import argparse
import queue
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
SAMPLE_RATE = 16_000
DEFAULT_SILENCE_THRESHOLD = 0.5  # seconds
DEFAULT_MIN_SPEECH_DURATION = 0.3  # seconds
DEFAULT_MAX_PHRASE_DURATION = 30.0  # seconds
VAD_SPEECH_THRESHOLD = 0.5  # probability threshold
VAD_WINDOW_SIZE = 512  # silero-vad expects exactly 512 samples at 16kHz
MAX_QUEUE_SIZE = 10


class StreamingTranscriber:
    """Manages continuous audio capture and transcription with VAD-based phrase detection."""

    def __init__(
        self,
        silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
        min_speech_duration: float = DEFAULT_MIN_SPEECH_DURATION,
        max_phrase_duration: float = DEFAULT_MAX_PHRASE_DURATION,
        output_file: str = None,
        verbose: bool = False,
    ):
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.max_phrase_duration = max_phrase_duration
        self.output_file = output_file
        self.verbose = verbose
        self.sample_rate = SAMPLE_RATE

        # State management
        self.recording = False
        self.running = True
        self.lock = threading.Lock()

        # Audio buffers and queues
        self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.buffer = []  # Phrase buffer for transcription
        self.vad_buffer = []  # Window buffer for VAD processing

        # VAD state tracking
        self.speech_active = False
        self.last_speech_time = 0
        self.phrase_start_time = 0
        self.silence_frames = 0
        self.speech_frames = 0

        # Model components (loaded once)
        self.model = None
        self.processor = None
        self.vad_model = None

        # Statistics
        self.phrases_processed = 0
        self.phrases_dropped = 0
        self.start_time = None

    def load_models(self):
        """Load Voxtral and VAD models once at startup."""
        print("[INIT] Loading VAD model...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        self.vad_model.eval()
        print("[INIT] VAD model loaded.")

        print("[INIT] Loading Voxtral model and processor...")
        self.model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID)
        self.processor = VoxtralProcessor.from_pretrained(MODEL_ID)
        print("[INIT] Voxtral model loaded successfully.")

        # Warmup: run a dummy inference to avoid cold start latency
        print("[INIT] Warming up model...")
        dummy_audio = np.zeros(self.sample_rate * 2, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, dummy_audio, self.sample_rate)
            try:
                self._transcribe_chunk(tmp.name)
            except Exception:
                pass  # Warmup may fail, that's ok
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        print("[INIT] Warmup complete.\n")

    def audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"[WARN] Audio callback status: {status}", file=sys.stderr)

        with self.lock:
            if not self.recording:
                return

            # Flatten audio chunk
            audio_chunk = indata.flatten()

            # Always add to phrase buffer (for later transcription)
            self.buffer.append(audio_chunk.copy())

            # Add to VAD buffer
            self.vad_buffer.append(audio_chunk.copy())

            vad_window_samples = sum(len(chunk) for chunk in self.vad_buffer)
            if vad_window_samples >= VAD_WINDOW_SIZE:
                all_vad_audio = np.concatenate(self.vad_buffer)
                processed = 0

                while processed + VAD_WINDOW_SIZE <= len(all_vad_audio):
                    vad_audio = all_vad_audio[processed : processed + VAD_WINDOW_SIZE].astype(np.float32, copy=False)
                    audio_tensor = torch.from_numpy(vad_audio)

                    try:
                        with torch.no_grad():
                            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
                    except Exception as exc:
                        if self.verbose:
                            print(f"[WARN] VAD inference skipped: {exc}", file=sys.stderr)
                        return

                    current_time = time.time()

                    if speech_prob > VAD_SPEECH_THRESHOLD:
                        if not self.speech_active:
                            self.speech_active = True
                            self.phrase_start_time = current_time
                            if self.verbose:
                                print("[VAD] Speech started", file=sys.stderr)

                        self.last_speech_time = current_time
                        self.speech_frames += VAD_WINDOW_SIZE
                        self.silence_frames = 0
                    elif self.speech_active:
                        self.silence_frames += VAD_WINDOW_SIZE
                        silence_duration = self.silence_frames / self.sample_rate

                        if silence_duration >= self.silence_threshold:
                            phrase_duration = self.speech_frames / self.sample_rate

                            if phrase_duration >= self.min_speech_duration:
                                phrase_audio = np.concatenate(self.buffer)
                                try:
                                    self.audio_queue.put_nowait(phrase_audio)
                                    if self.verbose:
                                        print(f"[VAD] Phrase complete ({phrase_duration:.2f}s) -> queue", file=sys.stderr)
                                except queue.Full:
                                    self.phrases_dropped += 1
                                    if self.verbose:
                                        print("[WARN] Queue full, dropping phrase", file=sys.stderr)
                            elif self.verbose:
                                print(f"[VAD] Phrase too short ({phrase_duration:.2f}s) -> discarded", file=sys.stderr)

                            self.buffer = []
                            self.speech_active = False
                            self.speech_frames = 0
                            self.silence_frames = 0

                    processed += VAD_WINDOW_SIZE

                remainder = all_vad_audio[processed:]
                self.vad_buffer = [remainder.copy()] if len(remainder) > 0 else []

            # Safety check: enforce max phrase duration
            if self.speech_active:
                current_time = time.time()
                phrase_duration = (current_time - self.phrase_start_time)
                if phrase_duration >= self.max_phrase_duration:
                    # Force chunk even if still speaking
                    phrase_audio = np.concatenate(self.buffer)
                    try:
                        self.audio_queue.put_nowait(phrase_audio)
                        if self.verbose:
                            print(f"[VAD] Max duration reached ({phrase_duration:.2f}s) → forced chunk", file=sys.stderr)
                    except queue.Full:
                        self.phrases_dropped += 1

                    self.buffer = []
                    self.vad_buffer = []
                    self.speech_active = False
                    self.speech_frames = 0
                    self.silence_frames = 0

    def _transcribe_chunk(self, audio_path: str) -> str:
        """Transcribe a single audio chunk."""
        inputs = self.processor.apply_transcrition_request(
            language="en",
            audio=audio_path,
        )

        # Convert TranscriptionInputs → mapping
        model_inputs = inputs
        if not isinstance(model_inputs, Mapping):
            if hasattr(model_inputs, "to_dict"):
                model_inputs = model_inputs.to_dict()
            else:
                model_inputs = vars(model_inputs)

        outputs = self.model.generate(**model_inputs, max_new_tokens=1024)

        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("Missing input_ids in transcription inputs.")

        prompt_len = input_ids.shape[1]
        return self.processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    def transcription_worker(self):
        """Worker thread that processes audio phrases from the queue."""
        while self.running:
            try:
                # Wait for audio phrase with timeout to allow clean shutdown
                phrase = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Save phrase to temporary wav file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, phrase, self.sample_rate)

            try:
                transcript = self._transcribe_chunk(tmp_path)
                transcript = transcript.strip()

                if transcript:
                    elapsed = time.time() - self.start_time
                    timestamp = time.strftime("%M:%S", time.gmtime(elapsed))
                    output = f"[{timestamp}] {transcript}"

                    print(output)

                    if self.output_file:
                        with open(self.output_file, "a") as f:
                            f.write(output + "\n")

                    self.phrases_processed += 1

                    if self.verbose:
                        queue_size = self.audio_queue.qsize()
                        print(f"[STATS] Phrases: {self.phrases_processed} | Queue: {queue_size}/{MAX_QUEUE_SIZE} | Dropped: {self.phrases_dropped}", file=sys.stderr)

            except Exception as e:
                print(f"[ERROR] Transcription failed: {e}", file=sys.stderr)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
                self.audio_queue.task_done()

    def toggle_recording(self):
        """Toggle recording state on/off."""
        with self.lock:
            self.recording = not self.recording
            state = "●" if self.recording else "○"
            status = "Recording" if self.recording else "Paused"
            print(f"\n{state} {status}")

            if not self.recording:
                # Clear buffers when pausing
                self.buffer = []
                self.vad_buffer = []
                self.speech_active = False
                self.speech_frames = 0
                self.silence_frames = 0

    def command_worker(self):
        """Command loop: Enter/toggle, pause, start, quit."""
        while self.running:
            try:
                command = input().strip().lower()
            except EOFError:
                return

            if command in {"", "t", "toggle"}:
                self.toggle_recording()
            elif command in {"s", "start"}:
                if not self.recording:
                    self.toggle_recording()
            elif command in {"p", "pause", "stop"}:
                if self.recording:
                    self.toggle_recording()
            elif command in {"q", "quit", "exit"}:
                print("\n[EXIT] Stopping...")
                self.running = False
                return

    def run(self):
        """Main entry point: start all threads and audio stream."""
        self.load_models()
        self.start_time = time.time()

        print("=" * 60)
        print("STREAMING SPEECH-TO-TEXT (VAD-based)")
        print("=" * 60)
        print(f"Silence threshold: {self.silence_threshold}s")
        print(f"Min speech duration: {self.min_speech_duration}s")
        print(f"Max phrase duration: {self.max_phrase_duration}s")
        print(f"Sample rate: {self.sample_rate} Hz")
        if self.output_file:
            print(f"Output file: {self.output_file}")
        if self.verbose:
            print(f"Verbose mode: ON")
        print("\nControls:")
        print("  ENTER / t / toggle  - Toggle recording on/off")
        print("  s / start           - Start recording")
        print("  p / pause           - Pause recording")
        print("  q / quit            - Stop and exit")
        print("=" * 60)
        print("\n○ Paused (press Enter to start)\n")

        # Start transcription worker thread
        worker = threading.Thread(target=self.transcription_worker, daemon=True)
        worker.start()

        # Start command input worker (avoids global input monitoring requirements)
        command_thread = threading.Thread(target=self.command_worker, daemon=True)
        command_thread.start()

        # Start audio stream
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self.audio_callback,
            ):
                # Keep main thread alive
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[EXIT] Interrupted")
        finally:
            self.running = False

            # Wait for queue to drain
            if not self.audio_queue.empty():
                print("\n[EXIT] Processing remaining phrases...")
                self.audio_queue.join()

            worker.join(timeout=5)
            command_thread.join(timeout=1)

            print(f"\n[STATS] Processed: {self.phrases_processed} phrases")
            if self.phrases_dropped > 0:
                print(f"[STATS] Dropped: {self.phrases_dropped} phrases")
            elapsed = time.time() - self.start_time
            print(f"[STATS] Session duration: {time.strftime('%M:%S', time.gmtime(elapsed))}")
            print("[EXIT] Done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=DEFAULT_SILENCE_THRESHOLD,
        help=f"Silence duration to trigger phrase end (default {DEFAULT_SILENCE_THRESHOLD}s)",
    )
    parser.add_argument(
        "--min-speech-duration",
        type=float,
        default=DEFAULT_MIN_SPEECH_DURATION,
        help=f"Minimum phrase duration to transcribe (default {DEFAULT_MIN_SPEECH_DURATION}s)",
    )
    parser.add_argument(
        "--max-phrase-duration",
        type=float,
        default=DEFAULT_MAX_PHRASE_DURATION,
        help=f"Maximum phrase duration before forcing chunk (default {DEFAULT_MAX_PHRASE_DURATION}s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file for transcripts (optional)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output with VAD and stats info",
    )
    args = parser.parse_args()

    transcriber = StreamingTranscriber(
        silence_threshold=args.silence_threshold,
        min_speech_duration=args.min_speech_duration,
        max_phrase_duration=args.max_phrase_duration,
        output_file=args.output,
        verbose=args.verbose,
    )
    transcriber.run()


if __name__ == "__main__":
    main()
