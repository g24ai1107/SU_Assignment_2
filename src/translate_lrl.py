INPUT_PATH = "outputs/ipa_output.txt"
OUTPUT_PATH = "outputs/lrl_output.txt"

# Normalization dictionary (instead of LRL translation)
LRL_DICT = {
    "data": "data",
    "model": "model",
    "probability": "probability",
    "lecture": "lecture",
    "bayes": "bayes",
    "stochastic": "stochastic",
    "hai": "hai",
    "is": "is",
    "the": "the"
}

def translate(text):
    words = text.split()
    translated = []

    for word in words:
        translated.append(LRL_DICT.get(word, word))

    return " ".join(translated)

def main():
    with open(INPUT_PATH, "r") as f:
        text = f.read()

    output = translate(text)

    with open(OUTPUT_PATH, "w") as f:
        f.write(output)

    print("Processing done!")
    print(output[:200])

if __name__ == "__main__":
    main()