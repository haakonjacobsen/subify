import whisper

def main():
    model = whisper.load_model("base")
    filename = "<INSERT-FILENAME>"
    audio = whisper.load_audio(filename)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)

if __name__ == "__main__":
    main()