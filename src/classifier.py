from Guesserator import guesserate
from youtubeSupport import youtube_get
import random
def run_comparison(function):
    pass

def random_guess(wav):
    g = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    return random.choice(g),(1 / len(g))

def main():
    print("Welcome to the Genre Classifier!")
    link = input("Enter a youtube link: ")

    #call link-to-wav fcn
    wav = youtube_get(link)
    methods = [("CRNN",guesserate), ("Random",random_guess)]
    for name, m in methods:
        (genre, certainty) = m(wav)
        print("{}: Song is {} with {}% certainty".format(name, genre, int(100*certainty)))
if __name__ == "__main__":
    main()
