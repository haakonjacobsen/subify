import torchaudio

def extract_audio(filename, start, end):
    """Extracts audio from a file between the start and end times.

    Args:
        filename (str): Path to the audio file.
        start (float): Start time in seconds.
        end (float): End time in seconds.

    Returns:
        torch.Tensor: Audio waveform.
    """
    audio, _ = torchaudio.load(filename)
    start = int(start * 16000)
    end = int(end * 16000)
    audio = audio[:, start:end]
    return audio[:, start:end]