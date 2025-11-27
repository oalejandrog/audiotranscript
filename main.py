import pyaudio
import wave
from faster_whisper import WhisperModel
import datetime
import re
import os
import sys
import argparse
import time
import threading
import queue
import numpy as np
import torch
import webrtcvad
import collections
import logging
from contextlib import contextmanager
from benchmarking import Benchmark

# --- Configuration ---
OUTPUT_FOLDER = "output_transcription"
RATE = 16000  # Sample rate
CHUNK = 1024  # Size of each audio chunk
CHANNELS = 1  # Mono audio
FORMAT = pyaudio.paInt16
# Time in seconds for the audio buffer used for transcription
TRANSCRIPTION_BUFFER_SECONDS = 1  # We'll use a smaller buffer and VAD
# Threshold for audio volume to be considered "not silence"
SILENCE_THRESHOLD = 0.01

# Common phrases to hint the model (Cache/Optimization)
COMMON_PHRASES = "Hello, how are you? Thank you. Please. Yes. No. Goodbye. I understand. Can you help me?"

# --- Globals ---
audio_queue = queue.Queue()
speech_queue = queue.Queue()
recording_stop_event = threading.Event()

# Silence faster_whisper logs
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

@contextmanager
def ignore_stderr():
    """Context manager to suppress C-level stderr output (ALSA warnings)."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
    except Exception:
        # If anything goes wrong with suppression, just yield
        yield

class VADAudio:
    """A class to wrap PyAudio and WebRTC VAD."""
    def __init__(self, aggressiveness=3, frame_duration_ms=30):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = RATE
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0) * 2)
        self.ring_buffer = collections.deque(maxlen=30) # buffer for context
        self.triggered = False

    def vad_collector(self, in_data):
        """Generator that yields audio frames from a buffer."""
        for frame in self.frames_from_buffer(in_data):
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            
            if not self.triggered:
                self.ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in self.ring_buffer if speech])
                if num_voiced > 0.9 * self.ring_buffer.maxlen:
                    self.triggered = True
                    yield b''.join([f for f, s in self.ring_buffer])
                    self.ring_buffer.clear()
            else:
                yield frame
                self.ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                    self.triggered = False
                    yield None
                    self.ring_buffer.clear()

    def frames_from_buffer(self, in_data):
        """Generator that yields audio frames from a buffer."""
        offset = 0
        while offset + self.frame_size <= len(in_data):
            yield in_data[offset:offset + self.frame_size]
            offset += self.frame_size


def record_thread(chunk, rate, channels, format):
    """
    Captures audio from the microphone and puts it into a queue.
    """
    # Suppress ALSA errors during initialization
    with ignore_stderr():
        p = pyaudio.PyAudio()
        
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    
    print("Recording started. Press Ctrl+C to stop.")
    
    while not recording_stop_event.is_set():
        try:
            data = stream.read(chunk, exception_on_overflow=False)
            audio_queue.put(data)
        except Exception as e:
            print(f"Error reading from audio stream: {e}")
            break
            
    print("Recording thread stopping.")
    stream.stop_stream()
    stream.close()
    p.terminate()

def vad_thread(vad_aggressiveness):
    """
    Consumes raw audio from audio_queue, performs VAD, and pushes complete speech segments to speech_queue.
    """
    print("VAD processing started.")
    vad_audio = VADAudio(aggressiveness=vad_aggressiveness)
    frames = b''

    while not recording_stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=1)
            
            for chunk in vad_audio.vad_collector(audio_data):
                if chunk is not None:
                    frames += chunk
                else: # End of speech detected
                    if frames:
                        # Push complete speech segment to inference queue
                        speech_queue.put(frames)
                        # print(f"[VAD] Speech segment detected ({len(frames)} bytes), queued for transcription.")
                    frames = b'' # Reset buffer

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in VAD thread: {e}")
            break
    
    print("VAD thread stopping.")

def transcribe_thread(model_name, device, language, beam_size, compute_type_arg):
    """
    Transcribes audio segments from the speech_queue using the Whisper model.
    """
    print(f"Loading Faster-Whisper model: {model_name}...")
    model = None

    if device == "cuda":
        if compute_type_arg != "auto":
            # User forced a specific type
            try:
                print(f"Attempting to load on {device} with {compute_type_arg} precision (forced)...")
                model = WhisperModel(model_name, device=device, compute_type=compute_type_arg)
                print(f"Success: Model loaded on {device} with {compute_type_arg} precision.")
            except Exception as e:
                print(f"Failed to load on {device} with {compute_type_arg}: {e}")
        else:
            # Auto-detect: Try preferred compute types for CUDA
            # GTX 10xx series often doesn't support native float16 efficient execution.
            # We try a sequence of fallbacks.
            compute_types_to_try = ["float16", "bfloat16", "int8_float16", "int8", "float32"]
            
            for ct in compute_types_to_try:
                try:
                    print(f"Attempting to load on {device} with {ct} precision...")
                    model = WhisperModel(model_name, device=device, compute_type=ct)
                    print(f"Success: Model loaded on {device} with {ct} precision.")
                    break
                except Exception as e:
                    print(f"Failed to load on {device} with {ct}: {e}")

    # Final fallback to CPU if CUDA failed or wasn't requested
    if model is None:
        print("Falling back to CPU (int8)...")
        try:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            print("Success: Model loaded on CPU with int8 precision.")
        except Exception as e:
            print(f"CRITICAL Error: Could not load model on CPU. {e}")
            return

    print(f"Transcriber ready (Language: {language}, Beam Size: {beam_size}).")

    while not recording_stop_event.is_set():
        try:
            # Wait for speech segments
            frames = speech_queue.get(timeout=1)
            
            # Convert buffer to numpy array
            audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe with Benchmark
            with Benchmark("Transcription Segment"):
                # Streaming generator - iterating triggers inference
                segments, info = model.transcribe(
                    audio_np, 
                    beam_size=beam_size,
                    language=language,
                    initial_prompt=COMMON_PHRASES
                )
                
                # Process segments as they stream in
                full_text = []
                for segment in segments:
                    full_text.append(segment.text)
                
                text = " ".join(full_text).strip()

            if text:
                print(f"Transcription: {text}")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in transcription thread: {e}")
            break
    
    print("Transcription thread stopping.")

def main():
    parser = argparse.ArgumentParser(description="Real-time Audio Recorder with Faster-Whisper Transcription and VAD")
    parser.add_argument("--prep-time", type=int, default=3, help="Preparation time before recording starts in seconds.")
    parser.add_argument("--model", type=str, default="small", help="Whisper model to use (e.g., tiny, base, small, medium, large).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation ('cpu' or 'cuda').")
    parser.add_argument("--vad-aggressiveness", type=int, default=3, choices=range(4), help="Set VAD aggressiveness from 0 to 3 (3 is most aggressive).")
    parser.add_argument("--language", type=str, default="en", help="Language code for transcription (e.g., 'en'). Default is 'en'. Set to None to enable auto-detection (slower).")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size for decoding. Default is 1 (greedy) for speed. Increase for accuracy.")
    parser.add_argument("--compute-type", type=str, default="auto", help="Compute type for model (e.g., float16, int8, int8_float16, float32). Default 'auto' checks availability.")
    args = parser.parse_args()
    
    # Handle "None" string from CLI if user really wants auto-detection
    lang = args.language if args.language.lower() != "none" else None

    if args.prep_time > 0:
        print(f"Prepare to speak...")
        for i in range(args.prep_time, 0, -1):
            print(f"Recording will start in {i} seconds...   ", end="\r")
            time.sleep(1)
        print("Recording will start now!            ")

    # Start the recording thread
    recorder = threading.Thread(target=record_thread, args=(CHUNK, RATE, CHANNELS, FORMAT))
    recorder.start()

    # Start the VAD thread
    vad_processor = threading.Thread(target=vad_thread, args=(args.vad_aggressiveness,))
    vad_processor.start()

    # Start the transcription thread
    transcriber = threading.Thread(target=transcribe_thread, args=(args.model, args.device, lang, args.beam_size, args.compute_type))
    transcriber.start()

    try:
        while recorder.is_alive() and transcriber.is_alive() and vad_processor.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping application...")
        recording_stop_event.set()

    # Wait for threads to finish
    recorder.join()
    vad_processor.join()
    transcriber.join()
    print("Application stopped.")


if __name__ == "__main__":
    main()
