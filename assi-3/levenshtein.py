import nltk

nltk.download('punkt')

def levenshtein_distance(a, b):
    if a == b: return 0
    if len(a) == 0: return len(b)
    if len(b) == 0: return len(a)

    v0 = [i for i in range(len(b) + 1)]
    v1 = [0] * (len(b) + 1)

    for i in range(len(a)):
        v1[0] = i + 1
        for j in range(len(b)):
            cost = 0 if a[i] == b[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0 = v1[:]

    return v1[len(b)]

def calculate_levenshtein_distance(df, text_col1, text_col2):
    distances = []
    for _, row in df.iterrows():
        tokenized_a = nltk.word_tokenize(row[text_col1])
        tokenized_b = nltk.word_tokenize(row[text_col2])
        distances.append(levenshtein_distance(tokenized_a, tokenized_b))
    return distances
