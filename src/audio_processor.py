import whisper
import json
import os
from pyannote.audio import Pipeline

class AudioProcessor:
    def __init__(self, model_size="base", device="auto"):
        """
        Initialize Whisper and Pyannote diarization.
        """
        if device == "auto":
            try:
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize Whisper
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size, device=str(self.device))
        print("Whisper model loaded.")
        
        # Initialize Pyannote diarization
        print("Loading Pyannote diarization model...")
        try:
            # You need a HuggingFace token
            hf_token = os.getenv("HF_TOKEN") or "your_huggingface_token_here"
            self.diarizer = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            if str(self.device) == "cuda":
                self.diarizer.to(torch.device("cuda"))
            print("Pyannote diarization model loaded.")
        except Exception as e:
            print(f"Failed to load Pyannote: {e}")
            print("Using mock diarizer instead.")
            self.diarizer = None

    def process_audio(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # 1. Diarization
        print(f"Diarizing {file_path}...")
        senko_segments = []
        
        if self.diarizer:
            try:
                diarization = self.diarizer(file_path)
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    senko_segments.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })
            except Exception as e:
                print(f"Diarization failed: {e}")
                senko_segments = [{"start": 0.0, "end": 60.0, "speaker": "Unknown"}]
        else:
            # Mock diarization
            senko_segments = [
                {"start": 0.0, "end": 15.0, "speaker": "Agent"},
                {"start": 15.0, "end": 30.0, "speaker": "Customer"},
                {"start": 30.0, "end": 45.0, "speaker": "Agent"}
            ]

        # 2. Transcription
        print(f"Transcribing {file_path}...")
        whisper_result = self.model.transcribe(file_path)

        # 3. Merge segments
        diarized_transcript = []
        for seg in whisper_result["segments"]:
            mid_time = (seg["start"] + seg["end"]) / 2
            speaker_label = "Unknown"
            
            for s in senko_segments:
                if s["start"] <= mid_time <= s["end"]:
                    speaker_label = s["speaker"]
                    break
            
            diarized_transcript.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": speaker_label,
                "text": seg["text"].strip()
            })

        return diarized_transcript
    
    # ... rest of the class ...