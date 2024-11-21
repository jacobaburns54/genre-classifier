
def main():
    print("Welcome to the Genre Classifier!")
    link = input("Enter a youtube link: ")

    #call link-to-wav fcn
    wav = None

    #call wav-to-genre fcn
    genre = None

    print("The genre of this song is " + genre + "!")

if __name__ == "__main__":
    main()