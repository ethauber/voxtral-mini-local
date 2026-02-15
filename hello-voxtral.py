import pathlib
from collections.abc import Mapping

from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
AUDIO_PATH = "hello.wav"

def main():
    if not pathlib.Path(AUDIO_PATH).exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    print("Loading model and processor...")
    model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID)
    processor = VoxtralProcessor.from_pretrained(MODEL_ID)

    print("Processing audio...")
    # Using the exact library method (typo included!)
    # Passing the file path directly handles the 16kHz resampling for you
    inputs = processor.apply_transcrition_request(
        language="en",
        audio=AUDIO_PATH
    )

    print("Generating transcript...")
    model_inputs = inputs
    if not isinstance(model_inputs, Mapping):
        if hasattr(model_inputs, "to_dict"):
            model_inputs = model_inputs.to_dict()
        else:
            model_inputs = vars(model_inputs)

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )

    # Calculate where the prompt ends and the generated text begins
    input_ids = (
        model_inputs.get("input_ids")
        if isinstance(model_inputs, Mapping)
        else getattr(inputs, "input_ids", None)
    )
    if input_ids is None:
        raise ValueError("Missing input_ids in transcription inputs.")

    prompt_len = input_ids.shape[1]

    transcript = processor.decode(
        outputs[0][prompt_len:],
        skip_special_tokens=True
    )

    print("\n--- Transcript ---")
    print(transcript)

if __name__ == "__main__":
    main()