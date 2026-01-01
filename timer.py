import time
import pygame
import os

# File name of the music
MUSIC_FILE = "C://Users//uaser/Desktop//song.mp3"  # Change this to your actual file name

# Check if the file exists
if not os.path.exists(MUSIC_FILE):
    print(f"Error: '{MUSIC_FILE}' not found in the current directory.")
    exit()

# Wait for 6 seconds
print("Waiting 6 seconds before playing music...")
time.sleep(6)

# Initialize pygame mixer
pygame.mixer.init()

# Load and play the music
pygame.mixer.music.load(MUSIC_FILE)
pygame.mixer.music.play()

print(f"Now playing: {MUSIC_FILE}")

# Keep the program running while music is playing
while pygame.mixer.music.get_busy():
    time.sleep(1)
