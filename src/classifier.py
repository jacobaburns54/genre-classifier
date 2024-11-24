from Guesserator import guesserate
from youtubeSupport import youtube_get
import random
import time
import os
import tracemalloc
def run_comparison(function):
    pass

def random_guess(wav):
    g = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    return random.choice(g),(1 / len(g))

def main():
    while True:
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
            tracemalloc.start()
            (genre, certainty) = m(wav)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            print("\n{} is {}% certain song is of genre {}.".format(name, int(100*certainty), genre))
            print("Took {}s and used {}MB of memory.\n".format(round(time.time()-start,3), round(peak / (1024 ** 2),3)))
        os.remove(wav)
        if input("Run again? (Y/n): ").lower() == "n":
            break
if __name__ == "__main__":
    main()
