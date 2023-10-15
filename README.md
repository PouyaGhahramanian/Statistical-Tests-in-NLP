# Statistical Tests in NLP (Bilkent EEE586 Assignment III)
## "Statistical Tests"
### Assignment Description
In this homework, you will find collocations in bigram form with 3 hypothesis testing methods. These are student’s t-test, chi-square test and likelihood ratio test. You will use the corpus in “Fyodor Dostoyevski Processed.txt” that is provided. This corpus consists of the concatenation of six novels of Fyodor Dostoyevski: Crime and Punishment, Devils, Notes from the Underground, The Brothers Karamazov, The Gambler and The Idiot.
### Part 1: Corpus Preprocessing
In the first part of the assignment, we aim to identify, analyze, and save collocations from the text corpus. I used the following steps in this part.

    Text processing: The text is read, tokenized, POS-tagged, and lemmatized using NLTK functions and a custom lemmatizer.

    Collocation finding: Bigram collocations are identified using the find-collocations function. This function uses an object of the BigramCollocationFinder from the NLTK library, and applies frequency and word filters to exclude certain bigrams.

    Dataframe creation: The get_collocations_dataframe function creates pandas dataframes for the collocations. These dataframes contain the individual words in each bigram, their frequencies, and the bigram frequencies.

    Saving results: Word frequencies are saved to a pickle file, and the dataframes are saved to CSV files.

The complete code for this part is given in the appendix as p1.py.
### Part 2: Finding the Collocations

In this part, we conduct statistical analysis on the collocations. The complete code for this part is given in the appendix as p2.py, and it consists of the following steps:

    Loading data: First, it read two CSV files containing the collocation data and a pickle file containing the word frequencies, all of which were created in the previous part.
    
    Statistical tests: It defines functions to calculate the t-score, chi-square score, and likelihood ratio for each row (bigram) in the dataframes. These calculations are based on the observed and expected frequencies of the words and bigrams.
    
    Applying tests: It applies the test functions to each row in the dataframes, creating new columns for the results.
    
    Saving results: It saves the updated dataframes with the score columns to new CSV files, collocations1_scores.csv and collocations3_scores.csv, one for each window size.
    
    Printing top scores: Finally, it sorts the dataframes by each score in turn, and prints the top 20 bigrams for each score and window size. The results for each test and window size are available in the answer sheet.
    
### Part 3: Explaining the Statistical Tests

#### Part (a)

I obtained the scores for all bigrams in part 2 and saved them to a pandas dataframe (collocations1_scores.csv and collocations3_scores.csv for two window sizes). Then I obtained the scores for the given bigrams by locating them in the pandas dataframe.

    t-score: The t-score is calculated as follows:

    t = (O11 - E11) / sqrt(O11)

    where O11 is the observed frequency of the bigram and E11 is the expected frequency, calculated as:

    E11 = (R1 * C1) / N

    R1 is the frequency of the first word in the corpus, C1 is the frequency of the second word, and N is the total number of bigrams.

    Chi-square score: The chi-square score is calculated using the formula:

    X^2 = N * (O11O22 - O12O21)^2 / ((O11 + O12) * (O11 + O21) * (O12 + O22) * (O21 + O22))

    O12 is the frequency of the first word without the second, O21 is the frequency of the second word without the first, and O22 is the frequency of neither the first word nor the second word occurring.

    Log-likelihood score: The log-likelihood scoreis calculated as:

    G^2 = 2 * [(O11 * log(O11/E11)) + (O12 * log(O12/E12)) + (O21 * log(O21/E21)) + (O22 * log(O22/E22))]

    Here, E12, E21, and E22 are the expected frequencies for O12, O21, and O22 respectively, calculated in a similar manner to E11.

#### Part (b)

We use the following procedure for each test to decide whether the bigrams are collocations or not. The obtained scores and threshold values are given in the answer sheet.

    For the t-score, we compare the calculated t-score with the critical t-value for α = 0.005 in the t-distribution table. If the absolute value of the t-score is greater than the critical value, we reject the null hypothesis that the words occur together by chance, and consider the bigram to be a collocation.
    
    For the chi-square test, we compare the calculated chi-square value with the critical chi-square value for α = 0.005 in the chi-square distribution table. If the chi-square value is greater than the critical value, we reject the null hypothesis and consider the bigram to be a collocation.
    
    For the log-likelihood test, we compare 2 times the calculated log-likelihood score with the critical chi-square value for α = 0.005 in the chi-square distribution table (because 2G^2 follows a chi-square distribution). If 2G^2 is greater than the critical value, we reject the null hypothesis and consider the bigram to be a collocation.
    

The degrees of freedom (DF) for the chi-square and log-likelihood tests in this case is 1, as we are considering bigrams (two-word combinations). The degrees of freedom for the t-test would be N-1 where N is the number of observations, which in this case would be quite large; therefore, we can use the standard normal distribution as an approximation.
