"""Record audio from microphone, then transcribe with Voxtral."""

import argparse
from collections.abc import Mapping

import sounddevice as sd
import soundfile as sf
from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
SAMPLE_RATE = 16_000  # Voxtral expects 16 kHz
DEFAULT_DURATION = 5  # seconds
DEFAULT_OUTPUT = "recording.wav"


def record_audio(duration: int, output_path: str) -> str:
    """Record from the default mic and save as 16 kHz mono WAV."""
    print(f"Recording {duration}s of audio (Ctrl-C to stop early)...")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    sf.write(output_path, audio, SAMPLE_RATE)
    print(f"Saved to {output_path}")
    return output_path


def transcribe(audio_path: str) -> str:
    """Load model, process audio, return transcript."""
    print("Loading model and processor...")
    model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID)
    processor = VoxtralProcessor.from_pretrained(MODEL_ID)

    print("Processing audio...")
    inputs = processor.apply_transcrition_request(
        language="en",
        audio=audio_path,
    )

    # Convert TranscriptionInputs → mapping so **unpacking works
    model_inputs = inputs
    if not isinstance(model_inputs, Mapping):
        if hasattr(model_inputs, "to_dict"):
            model_inputs = model_inputs.to_dict()
        else:
            model_inputs = vars(model_inputs)

    print("Generating transcript...")
    outputs = model.generate(**model_inputs, max_new_tokens=1024)

    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("Missing input_ids in transcription inputs.")

    prompt_len = input_ids.shape[1]
    return processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Recording length in seconds (default {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output WAV path (default {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    wav_path = record_audio(args.duration, args.output)
    transcript = transcribe(wav_path)
    print("\n--- Transcript ---")
    print(transcript)


if __name__ == "__main__":
    main()
