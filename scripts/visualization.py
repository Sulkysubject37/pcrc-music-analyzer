import matplotlib.pyplot as plt  
import librosa.display  
import numpy as np

class PCRCVisualizer:  
    @staticmethod  
    def plot_spectral_contrast(waveform, sr, track_name):  
        plt.figure(figsize=(12, 4))  
        S = librosa.feature.spectral_contrast(y=waveform, sr=sr)  
        librosa.display.specshow(S, x_axis='time')  
        plt.colorbar()  
        plt.title(f'Spectral Contrast - {track_name}')  
        return plt  

    @staticmethod  
    def plot_genre_radar(features_dict):  
        labels = list(features_dict.keys())  
        values = list(features_dict.values())  
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()  

        fig = plt.figure(figsize=(6, 6))  
        ax = fig.add_subplot(111, polar=True)  
        ax.plot(angles, values, color='#1DB954', linewidth=2)  # Spotify green  
        ax.fill(angles, values, alpha=0.25)  
        ax.set_yticklabels([])  
        ax.set_xticks(angles)  
        ax.set_xticklabels(labels, fontsize=8)  
        return plt  