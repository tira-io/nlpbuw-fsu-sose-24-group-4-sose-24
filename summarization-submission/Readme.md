1. Imports and Library Setup:

2. Main Function:
   - Initializes the TIRA client to load the dataset.

3. Generate Summaries:
   - Applies the `generate_summary` function to each story in the DataFrame to create summaries.
   - Adds the summaries as a new column named "summary".
   - Drops the original "story" column from the DataFrame.

4. Save Summaries:
   - Saves the DataFrame to a JSONL (JSON Lines) file.
   - Handles any exceptions that occur and prints an error message if needed.

5. Generate Summary Function:
   - Tokenizes the input text into sentences using `sent_tokenize`.
   - If the number of sentences is less than or equal to `num_sentences`, returns the original text.

6. TF-IDF and Cosine Similarity:
   - Converts sentences to a TF-IDF matrix.
   - Computes cosine similarity between the sentences.
   - Scores sentences based on their similarity to other sentences.
   - Ranks the sentences by their scores and selects the top `num_sentences` sentences.
   - Joins the selected sentences to form the summary.

7. Entry Point:
   - Ensures that the `main` function is called when the script is executed directly.












