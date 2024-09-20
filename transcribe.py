import whisper
import sounddevice as sd
import time
import numpy as np
import queue

# Two functions for time measurement and logging:
starttime = time.time()

def reltime():
    return time.time() - starttime

def log(message):
    print("\n[" + str(reltime()) + "] " + message);

model = whisper.load_model("base")

log("Initialization done")

# We're going to incrementally record audio from microphone,
# storing each chunk in the queue. It can be disabled setting
# dorec = False. This happens when CTRL+C is pressed and it's
# supposed only to process what is already recorded.
dorec = True

chunks = queue.Queue()

def push_audio_chunk(indata, frames, time, status):
    # Add the audio chunk to the queue
    if dorec == True:
        chunks.put(indata.copy())


# Procedure for processing queued audio chunks
SAMPLE_RATE = 16000  # Whisper works best with 16 kHz audio
CHUNK_DURATION = 5
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # Size of each chunk in samples

class InputProcessor:
    def __init__(self, model):
        self.buffer = np.array([], dtype=np.float32)
        self.model = model
        self.text = ""

    def accept(self, chunk):
        self.buffer = np.concatenate((self.buffer, chunk.flatten()))
        pos = len(self.buffer)
        if pos >= CHUNK_SIZE:
            #log("Running transcription @ " + str(pos))
            result = self.model.transcribe(self.buffer, initial_prompt=self.text, fp16=False)

            #log("Transcription finished: " + str(result))

            self.text = self.text + result["text"]
            log(result["text"])

            self.buffer = np.array([], dtype=np.float32)


def transcribe_audio(model):
    global dorec

    proc = InputProcessor(model)

    log("Transcription started (Ctrl+C to stop)...")

    while dorec == True or not(chunks.empty()):
        try:
            proc.accept(chunks.get())

        except KeyboardInterrupt:
            log("Transcription stopped. Still processing what's left in the buffer.")
            dorec = False
            pass

# Now, we start streaming from the microphone and run reading loop
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=push_audio_chunk, dtype='float32'):
    transcribe_audio(model)
