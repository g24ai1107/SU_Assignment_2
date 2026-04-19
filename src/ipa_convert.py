import json

INPUT_PATH = "outputs/transcript_constrained.json"
OUTPUT_PATH = "outputs/ipa_output.txt"

# Basic Hinglish → IPA mapping
IPA_MAP = {
    "namaste": "nəməsteː",
    "data": "deɪtə",
    "model": "mɒdəl",
    "probability": "prɒbəˈbɪlɪti",
    "bayes": "beɪz",
    "stochastic": "stoʊˈkæstɪk",
    "hai": "hɛ",
    "is": "ɪz",
    "the": "ðə",
    "lecture": "lɛktʃər"
}


def convert_to_ipa(text):
    words = text.lower().split()
    ipa_words = []

    for word in words:
        ipa_words.append(IPA_MAP.get(word, word))

    return " ".join(ipa_words)


def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    text = data["biased_text"]

    ipa_text = convert_to_ipa(text)

    with open(OUTPUT_PATH, "w") as f:
        f.write(ipa_text)

    print("IPA conversion done!")
    print("\nSample:\n")
    print(ipa_text[:200])


if __name__ == "__main__":
    main()