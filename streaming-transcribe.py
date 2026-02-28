"""Continuous streaming speech-to-text with toggle control."""

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
from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor
from pynput import keyboard

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
SAMPLE_RATE = 16_000
DEFAULT_CHUNK_DURATION = 5
MAX_QUEUE_SIZE = 10


class StreamingTranscriber:
    """Manages continuous audio capture and transcription with toggle control."""

    def __init__(
        self, chunk_duration: int = DEFAULT_CHUNK_DURATION, output_file: str = None
    ):
        self.chunk_duration = chunk_duration
        self.output_file = output_file
        self.sample_rate = SAMPLE_RATE

        # State management
        self.recording = False
        self.running = True
        self.lock = threading.Lock()

        # Audio buffers and queues
        self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.buffer = []

        # Model components (loaded once)
        self.model = None
        self.processor = None

        # Statistics
        self.chunks_processed = 0
        self.start_time = None

    def load_model(self):
        """Load Voxtral model and processor once at startup."""
        print("[INIT] Loading model and processor...")
        self.model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID)
        self.processor = VoxtralProcessor.from_pretrained(MODEL_ID)
        print("[INIT] Model loaded successfully.")

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

            # Accumulate audio data
            self.buffer.append(indata.copy())

            # Check if we have enough for a chunk
            total_frames = sum(len(chunk) for chunk in self.buffer)
            frames_needed = int(self.chunk_duration * self.sample_rate)

            if total_frames >= frames_needed:
                # Concatenate and push to queue
                chunk = np.concatenate(self.buffer)[:frames_needed]
                self.buffer = [np.concatenate(self.buffer)[frames_needed:]]

                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    print("[WARN] Queue full, dropping audio chunk", file=sys.stderr)

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
        """Worker thread that processes audio chunks from the queue."""
        while self.running:
            try:
                # Wait for audio chunk with timeout to allow clean shutdown
                chunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Save chunk to temporary wav file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, chunk, self.sample_rate)

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

                    self.chunks_processed += 1

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

    def on_press(self, key):
        """Keyboard event handler."""
        try:
            if key == keyboard.Key.space:
                self.toggle_recording()
            elif key == keyboard.Key.esc:
                print("\n[EXIT] Stopping...")
                self.running = False
                return False  # Stop listener
        except AttributeError:
            pass

    def run(self):
        """Main entry point: start all threads and audio stream."""
        self.load_model()
        self.start_time = time.time()

        print("=" * 60)
        print("STREAMING SPEECH-TO-TEXT")
        print("=" * 60)
        print(f"Chunk duration: {self.chunk_duration}s")
        print(f"Sample rate: {self.sample_rate} Hz")
        if self.output_file:
            print(f"Output file: {self.output_file}")
        print("\nControls:")
        print("  SPACE  - Toggle recording on/off")
        print("  ESC    - Stop and exit")
        print("=" * 60)
        print("\n○ Paused (press SPACE to start)\n")

        # Start transcription worker thread
        worker = threading.Thread(target=self.transcription_worker, daemon=True)
        worker.start()

        # Start keyboard listener
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

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
            listener.stop()

            # Wait for queue to drain
            if not self.audio_queue.empty():
                print("\n[EXIT] Processing remaining chunks...")
                self.audio_queue.join()

            worker.join(timeout=5)

            print(f"\n[STATS] Processed {self.chunks_processed} chunks")
            print("[EXIT] Done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--chunk-duration",
        type=int,
        default=DEFAULT_CHUNK_DURATION,
        help=f"Chunk duration in seconds (default {DEFAULT_CHUNK_DURATION})",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file for transcripts (optional)",
    )
    args = parser.parse_args()

    transcriber = StreamingTranscriber(
        chunk_duration=args.chunk_duration,
        output_file=args.output,
    )
    transcriber.run()


if __name__ == "__main__":
    main()
