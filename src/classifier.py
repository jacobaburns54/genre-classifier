from Guesserator import guesserate
from youtubeSupport import youtube_get
import random
import time
import os
def run_comparison(function):
    pass

def random_guess(wav):
    g = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    return random.choice(g),(1 / len(g))

def main():
    size = os.get_terminal_size()
    os.system("clear")
    print(" Welcome to the Genre Classifier! ".center(size.columns,"="))
    link = input("Enter a youtube link: ")
    print("\nLoading audio...")
    #call link-to-wav fcn
    wav = youtube_get(link)
    methods = [("CRNN",guesserate), ("Random",random_guess)]
    print("\nLoading done. Starting evaluation...")
    for name, m in methods:
        start = time.time()
        (genre, certainty) = m(wav)
        print("{}: Song is {} with {}% certainty. (took {}s)".format(name, genre, int(100*certainty), round(time.time()-start,3)))
    os.remove(wav)
if __name__ == "__main__":
    main()
