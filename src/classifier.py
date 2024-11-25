from Guesserator import guesserate
from youtubeSupport import youtube_get
import random
import time
import os
import tracemalloc
from RandomForest import random_forest

def run_comparison(function):
    pass

def random_guess(wav):
    g = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    return random.choice(g),(1 / len(g))

class Model:
    def __init__(self, name:str, func: callable):
        self.name = name
        self.func = func
        self.guesses = []
        self.confidences = []
        self.times = []
        self.mems = []
    def __repr__(self):
        return self.name
    def __call__(self, *args, **kwargs):
        start = time.time()
        tracemalloc.start()
        (genre, certainty) = self.func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_time = time.time() - start
        confidence = int(100 * certainty)

        self.confidences.append(confidence)
        self.times.append(total_time)
        self.mems.append(peak)
        print("\n{} Model is {}% certain that the song is {}.".format(self.name, confidence, genre))
        print("Took {} seconds and used {}MB of memory.\n".format(round(total_time,3), round(peak / (1024 ** 2), 3)))
        if input("Is that correct? (y/N): ").lower() == "y":
            self.guesses.append(1)
        else:
            self.guesses.append(0)

    def summary(self):
        acc = 0.0
        confidence = 0.0
        time = 0.0
        mem = 0.0
        for i in range(len(self.guesses)):
            acc += self.guesses[i]
            confidence += self.confidences[i]
            time += self.times[i]
            mem += self.mems[i]
        acc = int(100*acc / len(self.guesses))
        confidence = confidence / len(self.guesses)
        time = time / len(self.guesses)
        mem = mem / len(self.guesses)
        print("\n==={}===".format(self.name))
        print("Avg. Accuracy: {}%".format(acc))
        print("Avg. Confidence: {}%".format(round(confidence, 3)))
        print("Avg. Time: {}s".format(round(time,3)))
        print("Avg. Memory: {}MB".format(round(mem / (1024 ** 2),3)))

def main():
    while True:
        size = os.get_terminal_size()
        os.system("clear")
        print(" Welcome to the Genre Classifier! ".center(size.columns,"="))
        link = input("Enter a youtube link: ")
        print("\nLoading audio...")
        #call link-to-wav fcn
        wav = youtube_get(link)
        CRNN = Model("CRNN", guesserate)
        Random = Model("Random", random_guess)
        Forest = Model("Random Forest", random_forrest)
        methods: list[Model] = [CRNN, Forest, Random]
        print("\nLoading done. Starting evaluation...")
        for m in methods:
            m(wav)
        os.remove(wav)
        if input("Run again? (Y/n): ").lower() == "n":
            for m in methods:
                m.summary()
            break
if __name__ == "__main__":
    main()
