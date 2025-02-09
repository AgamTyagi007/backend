from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import whisper
import numpy as np
import soundfile as sf
import io
from pydub import AudioSegment
import uvicorn

app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model 
model = whisper.load_model("base")

@app.get("/")
def read_root():
    return {"message": "Whisper Noise Detection API is running."}

def convert_to_wav(audio_data: bytes, format: str):
    """
    Converts audio bytes to WAV using pydub.
    """
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=format)  
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        output_buffer = io.BytesIO()
        audio_segment.export(output_buffer, format="wav")  
        output_buffer.seek(0)
        return output_buffer
    except Exception as e:
        print(f"FFMPEG Conversion Error: {e}")
        return None

def check_if_noisy(audio_data: bytes, format: str):
    """
    Processes audio and determines if it's noisy.
    """
    print("Processing audio with Whisper...")

    # Convert to WAV format
    wav_file = convert_to_wav(audio_data, format)
    if not wav_file:
        raise HTTPException(status_code=400, detail="Invalid audio format")

    # Read WAV data into numpy array
    audio, sample_rate = sf.read(wav_file, dtype="float32")

    print("Audio Shape:", audio.shape) 

    # Transcribe with Whisper
    result = model.transcribe(audio)
    print("Whisper Output:", result)

    if "text" in result and result["text"].strip():
        transcription = result["text"].strip()
        segments = result.get("segments", [])

        is_noisy = False
        noisy_segments = []

        for segment in segments:
            no_speech_prob = segment.get("no_speech_prob", 0)
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            noisy_segments.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "no_speech_prob": no_speech_prob,
                }
            )
            if no_speech_prob > 0.5:
                is_noisy = True

        return {
            "transcription": transcription,
            "is_noisy": is_noisy,
            "noisy_segments": noisy_segments,
        }
    else:
        return {
            "transcription": None,
            "is_noisy": True,
            "message": "No speech detected. The audio might be too noisy or silent.",
        }

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    """
    Receives and processes audio files.
    """
    print(f"Received file: {file.filename}, Content-Type: {file.content_type}")
    audio_data = await file.read()  # Read file as bytes

    format = file.filename.split(".")[-1]  # Extract file extension
    print(f"File size: {len(audio_data)} bytes")

    # Validate format
    supported_formats = ["mp3", "wav", "ogg", "webm", "m4a"]
    if format.lower() not in supported_formats:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # Convert to WAV
    wav_buffer = convert_to_wav(audio_data, format)
    if not wav_buffer:
        return JSONResponse(content={"error": "Audio conversion failed"}, status_code=400)
    
    # Process with Whisper
    result = check_if_noisy(wav_buffer.getvalue(), "wav")
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
