1. Imports and Library Setup:

2. Main Function:
   - Initializes the TIRA client to load the dataset.
   - Applies the `generate_summary` function to each story in the DataFrame to create summaries.
   - Adds the summaries as a new column named "summary".
   - Drops the original "story" column from the DataFrame.

   - Saves the DataFrame to a JSONL (JSON Lines) file.
   - Handles any exceptions that occur and prints an error message if needed.

3. Preprocess Text Function:
   - Removes extra whitespace from the input text using a regular expression.
   - Tokenizes the text into sentences using `sent_tokenize`.
   - Returns the list of sentences.

4. Extract Keywords Function:
   - Loads the set of English stopwords.
   - Tokenizes the input text into words using `word_tokenize`.
   - Converts words to lowercase and filters out non-alphanumeric words and stopwords.
   - Counts the frequency of each word using `Counter`.
   - Extracts the top 10 most common words as keywords.
   - Returns the set of keywords.

5. Score Sentences Function:
   - Initializes a `defaultdict` to store sentence scores.
   - For each sentence, tokenizes it into words and converts to lowercase.
   - If a word is in the set of keywords, increments the score of the sentence.
   - Returns the dictionary of sentence scores.

6. Generate Summary Function:
   - Calls `preprocess_text` to get the list of sentences from the input text.
   - If the number of sentences is less than or equal to `num_sentences`, returns the original text.
   - Calls `extract_keywords` to get the set of keywords from the text.
   - Calls `score_sentences` to score the sentences based on the keywords.
   - Sorts the sentences by their scores in descending order.
   - Joins the top `num_sentences` sentences to form the summary.
   - Returns the summary.

7. Entry Point:
   - Ensures that the `main` function is called when the script is executed directly.
