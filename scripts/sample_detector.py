import acoustid
import chromaprint
from pathlib import Path

class SampleDetector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.fingerprints = {}

    def fingerprint_samples(self, sample_dir):
        """Fingerprint reference samples (Bollywood/RD Burman tracks)"""
        sample_dir = Path('data/rawtracks')
        for audio_file in sample_dir.glob('*.mp3'):
            duration, fingerprint = acoustid.fingerprint_file(str(audio_file))
            self.fingerprints[audio_file.name] = {
                'duration': duration,
                'fingerprint': fingerprint
            }

    def detect_sample(self, track_path):
        """Check if PCRC track contains known samples"""
        duration, track_fingerprint = acoustid.fingerprint_file(str(track_path))
        matches = []
        
        for sample_name, sample_data in self.fingerprints.items():
            score = acoustid.compare_fingerprints(
                track_fingerprint,
                sample_data['fingerprint']
            )
            if score > 0.5:  # Adjust threshold as needed
                matches.append({
                    'sample': sample_name,
                    'confidence': score
                })
        
        return matches