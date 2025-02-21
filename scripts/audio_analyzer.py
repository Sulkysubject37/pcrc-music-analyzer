import librosa  
import numpy as np  
import spotipy  
from spotipy.oauth2 import SpotifyClientCredentials  

class PCRCAnalyzer:  
    def __init__(self, track_path):  
        self.track_path = track_path  
        self.waveform, self.sr = librosa.load(track_path, duration=30)  

    def get_genre_tags(self):
        """Extract basic audio features to help determine genre"""
        mfccs = librosa.feature.mfcc(y=self.waveform, sr=self.sr, n_mfcc=13)
        spectral = librosa.feature.spectral_centroid(y=self.waveform, sr=self.sr)
        
        # Simple genre characteristics based on audio features
        features = {
            'brightness': np.mean(spectral),
            'complexity': np.std(mfccs),
            'energy': np.mean(librosa.feature.rms(y=self.waveform))
        }
        return features

    def get_spotify_features(self, client_id, client_secret):  
        """Fetch danceability/acousticness from Spotify API"""  
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))  
        results = sp.search(q=f'track:{self.track_path.stem}', type='track')  
        if results['tracks']['items']:  
            track_id = results['tracks']['items'][0]['id']  
            features = sp.audio_features(track_id)[0]  
            return {  
                'danceability': features['danceability'],  
                'acousticness': features['acousticness'],  
                'energy': features['energy']  
            }  
        return None  

    def get_custom_features(self):  
        """Extract PCRC-specific features (sitar presence, lo-fi noise)"""  
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=self.waveform, sr=self.sr))  
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(self.waveform))  
        return {  
            'spectral_centroid': spectral_centroid,  
            'zero_crossing_rate': zero_crossing,  
            'estimated_bpm': librosa.beat.tempo(y=self.waveform)[0]  
        }  