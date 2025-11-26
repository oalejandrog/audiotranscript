import pyaudio
import wave
import whisper
import datetime
import re
import os
import argparse
import time

OUTPUT_FOLDER = "output_transcription"

# Expanded list of hesitation markers, including common variations
HESITATION_MARKERS = [
    "um", "uh", "ah", "er", "well", "like", "you know", "so", "actually", "basically", 
    "I mean", "right", "okay", "just", "seriously", "mmm", "uhm", "hmm", "ahh", "eh", 
    "aah", "uhh", "huh", "let me think", "I guess", "I suppose", "I'm not sure", "I'm thinking", 
    "what I mean is", "it’s like", "you know what I mean", "I don’t know", "it’s kind of", 
    "kind of", "sort of", "something like that", "to be honest", "well, you see", "you see", 
    "I believe", "if you will", "literally", "honestly", "probably", "uhmm", "aahhh", "umm",
    # Spanish hesitation markers used by Spanish speakers speaking English
    "eh", "a ver", "bueno", "pues", "digamos", "como", "o sea", "ya sabes", "bueno, a ver", 
    "sabes", "ehhh", "mmm", "aaaa", "eeee", "mmmm", "mmm", "hmm"
]

def record_audio(record_seconds):
    print("Recording... Press Ctrl+C to stop.")
    
    RATE = 16000  # Sample rate
    CHUNK = 1024  # Size of each audio chunk
    FORMAT = pyaudio.paInt16  # Format for the audio input
    CHANNELS = 1  # Mono audio
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    AUDIO_FILE = os.path.join(OUTPUT_FOLDER, f"recorded_audio_{timestamp}.wav")
   
    # Ensure the output folder exists before recording
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    try:
        if record_seconds:
            num_chunks = int(RATE / CHUNK * record_seconds)
            for i in range(num_chunks):
                seconds_left = record_seconds - int(i * CHUNK / RATE)
                print(f"Recording... {seconds_left} seconds remaining", end="\r")
                data = stream.read(CHUNK)
                frames.append(data)
        else:
            while True:
                data = stream.read(CHUNK)
                frames.append(data)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    with wave.open(AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return AUDIO_FILE

def transcribe_audio(model, audio_path):
    result = model.transcribe(audio_path)
    return result['text']

def detect_hesitations(transcription):
    detected_hesitations = []

    # Normalize the transcription to lowercase to improve matching
    transcription = transcription.lower()

    for marker in HESITATION_MARKERS:
        if re.search(r'\b' + re.escape(marker) + r'\b', transcription):
            detected_hesitations.append(marker)

    return detected_hesitations

def save_transcription(transcription, audio_file, detected_hesitations):
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    transcript_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_transcript.txt")

    with open(transcript_path, 'w') as file:
        file.write("Transcription:\n")
        file.write(transcription)

        if detected_hesitations:
            file.write("\n\nDetected Hesitations:\n")
            file.write(", ".join(detected_hesitations))

    print(f"Transcript saved at {transcript_path}")

def main():
    parser = argparse.ArgumentParser(description="Audio Recorder with Whisper Transcription")
    parser.add_argument("--prep-time", type=int, default=0, help="Preparation time before recording in seconds")
    parser.add_argument("--seconds", type=int, default=None, help="Duration to record audio in seconds")
    parser.add_argument("--model", type=str, default="small", help="Whisper model to use (e.g., tiny, base, small, medium, large)")
    args = parser.parse_args()

    print(f"Loading Whisper model: {args.model}...")
    model = whisper.load_model(args.model)
    print("Model loaded.")

    print(f"Prepare to speak...", end="\n")
    if args.prep_time > 0:
        for i in range(args.prep_time, 0, -1):
            print(f"Recording will start in {i} seconds.", end="\r")
            time.sleep(1)
        print("Recording started!")

    audio_path = record_audio(args.seconds)
    transcription = transcribe_audio(model, audio_path)
    detected_hesitations = detect_hesitations(transcription)

    print(f"Transcription Preview: {transcription[:100]}...")

    save_transcription(transcription, audio_path, detected_hesitations)

if __name__ == "__main__":
    main()

