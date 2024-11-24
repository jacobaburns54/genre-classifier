from Guesserator import guesserate
from youtubeSupport import youtube_get
def run_comparison(function):
    pass

def main():
    print("Welcome to the Genre Classifier!")
    link = input("Enter a youtube link: ")

    #call link-to-wav fcn
    wav = youtube_get(link)

    #call wav-to-genre fcn
    (genre, certainty) = guesserate(wav)

    print("The genre of this song is " + genre + "!")
    print("With certainty: " + int(certainty*100) + "%!")

if __name__ == "__main__":
    main()
