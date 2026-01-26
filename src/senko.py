"""
Mock senko module for speaker diarization.
In production, replace with actual senko library if available.
"""

class Diarizer:
    def __init__(self, device="cpu", warmup=True, quiet=False):
        self.device = device
        print(f"Mock Senko Diarizer initialized on {device}")
    
    def diarize(self, file_path, generate_colors=False):
        """
        Returns mock diarization results for testing.
        In real usage, this would contain actual speaker diarization.
        """
        print(f"Mock diarization for: {file_path}")
        
        # Return mock segments with 2 speakers
        return {
            "merged_segments": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "speaker": "SPEAKER_01"
                },
                {
                    "start": 10.0,
                    "end": 20.0,
                    "speaker": "SPEAKER_02"
                },
                {
                    "start": 20.0,
                    "end": 30.0,
                    "speaker": "SPEAKER_01"
                }
            ]
        }