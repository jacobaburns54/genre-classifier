import subprocess
import sys
import os
from yt_dlp import YoutubeDL
#needs to be 30s, mono

def youtube_get(url: str):
    yt=YoutubeDL()
    path = yt.extract_info(url, download=False)['title']
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,  # Suppress all output
        'logger': None,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': path,  # Output file path
    }
    yt = YoutubeDL(ydl_opts)
    yt.download([url])
    trim_command = [
        'ffmpeg',
        '-i', path+".wav",  # Input file
        '-ss', '30',
        '-t', '60',  # Trim to 30 seconds
        '-ac', '1',  # Convert to mono
        '-loglevel', 'error',
        path+"-trimmed.wav"  # Output file
    ]
    try:
        # Run the command
        subprocess.run(trim_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing audio: {e}")
        return ""
    os.remove(path+".wav")
    return path+"-trimmed.wav"

if __name__ == "__main__":
    print(youtube_get(sys.argv[1]))