import whisper
import sounddevice as sd
import time

starttime = time.time()

# Load Whisper model
model = whisper.load_model("base")

# Define recording settings
SAMPLE_RATE = 16000  # Whisper works best with 16 kHz audio
DURATION = 30  # Record for 10 seconds

def reltime():
    return time.time() - starttime

def record_audio(duration, fs):
    """Record audio from the microphone."""
    print("Recording started(" + str(reltime()) + ")...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished(" + str(reltime()) + ")!")
    return audio.flatten()  # Flatten to 1D array

# Record audio from the microphone
audio_data = record_audio(DURATION, SAMPLE_RATE)

# Use Whisper to transcribe the audio
result = model.transcribe(audio_data, fp16=False)  # Use fp16=False if on CPU

# Print the transcription result
print("Transcribed text(" + str(reltime()) + "): \n" + result["text"])

