# Emoji Usage and Tone Analysis in Reddit Discussions

Are the skull (💀) and loudly crying face (😭) emojis used for their literal meanings, or for something else? This project analyzes 1,000,000 Reddit comments from May 2019 and finds that both emojis are overwhelmingly used figuratively, as markers of humor, exaggeration, and tone rather than literal death or sadness.

## Main Findings

Comments containing the emoji were labeled **literal** (real death, danger, sadness, or grief), **figurative** (hyperbole, humor, insult, endearment, or shock), or **unclear** (cannot tell). After removing spam and duplicates, 438 crying-face and 50 skull comments were usable.

| | Usable comments | Literal | Figurative | Unclear | Not literal (95% CI) |
|---|---|---|---|---|---|
| Crying 😭 | 438 | 42 (9.6%) | 359 (82.0%) | 37 (8.4%) | 90.4% (87.3% to 92.8%) |
| Skull 💀 | 50 | 0 (0%) | 47 (94.0%) | 3 (6.0%) | 100% (92.9% to 100%) |

- **Most uses are not literal.** None of the 50 skull comments used the skull for its literal meaning of death or danger.
- **Context does not always settle the meaning.** Even a careful reader marked 8.4% of crying comments and 6.0% of skull comments unclear.
- **An automated cross-check agrees in total but not comment by comment.** A sentence-embedding method flagged 93.4% of crying comments and 98.0% of skull comments as not literal at its preset threshold. For crying, it found only 8 of the 42 literal comments (precision 0.28, recall 0.19, F1 0.23).
- **A classifier trained on the reviewed labels does somewhat better.** For crying, TF-IDF with logistic regression (5-fold cross-validation, repeated 5 times) reached F1 0.39 (±0.03) for the literal class, compared with 0.00 for always guessing figurative. No classifier is possible for skull because it has no literal examples.

The reviewed labels are the main result. The embedding and the classifier are checks against them, and all of these numbers are exploratory because only 42 crying comments were literal.

## Repository Contents

| File | Description |
|---|---|
| `emoji_analysis.ipynb` | The full analysis notebook (Google Colab) |
| `cry_v2_labeled.csv` | Reviewed labels for the crying-face comments |
| `skull_v2_labeled.csv` | Reviewed labels for the skull comments |
| `README.md` | This file |
| `LICENSE` | MIT License |

The raw Reddit file (`kaggle_RC_2019-05.csv`, about 186 MB) is not included because of GitHub's file size limit. See "Dataset" below.

### Label file columns

| Column | Meaning |
|---|---|
| `subreddit`, `body` | The comment's community and text |
| `manual_sense` | The author's hand label (crying: 85 comments) |
| `claude_sense` | Label suggested by an LLM (Claude) for the remaining comments |
| `label_source` | `human` or `claude` |
| `final_sense` | The label used in the analysis (hand label if present, otherwise the suggested label) |
| `quality_note` | Flags for spam, duplicates, or non-English comments; flagged rows are excluded |
| `review_sample`, `human_check` | A random subset set aside for blind re-labeling, with a column to record the result |

**How the labels were made.** The author hand-labeled 85 crying-face comments. An LLM (Claude) labeled all the remaining crying-face comments and all the skull comments using the same rubric, and the author reviewed them. Because that review was not blind, these labels should be described as author-reviewed rather than independently hand-labeled.

## Dataset

This project uses the "1 Million Reddit Comments from 40 Subreddits" dataset from Kaggle:

https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits

1. Go to the link above and click "Download".
2. Extract the ZIP file.
3. Locate `kaggle_RC_2019-05.csv`. It should be about 185.9 MB and contain 1,000,000 rows.

## Methods

1. **Load and clean.** The raw file is read with pandas' Python parser, which correctly keeps comments that contain commas, line breaks, or quotation marks. Rows with malformed subreddit names are dropped and HTML entities are decoded.
2. **Find the emoji comments.** 504 comments contain 💀 or 😭. Spam (pasted emoji lists, copypasta), non-English comments, and duplicates are removed (16 comments), leaving 488.
3. **Describe usage.** Comments are grouped by subreddit category (sports, gaming, memes, politics/news, other), emoji type (skull, crying, or both), emoji intensity (repetition), and tone (serious, casual, or intensified emotion). An interactive ipywidgets dashboard lets you browse the results.
4. **Label meaning.** Each comment is labeled literal, figurative, or unclear (see "Label file columns" above).
5. **Embedding check.** With the emoji removed, each comment is embedded with `all-MiniLM-L6-v2` and compared with hand-written anchor phrases for the emoji's literal meaning (for example "death" and "a funeral" for 💀; "grief" and "heartbreak" for 😭). A comment is called ambiguous when its similarity falls below a preset threshold of 0.35.
6. **Validation.** The embedding is compared with the reviewed labels using confidence intervals (Wilson), precision, recall, F1, and a threshold sweep.
7. **Supervised check.** A TF-IDF and logistic regression classifier is trained on the reviewed crying labels (unclear comments excluded, emoji removed) and evaluated with repeated cross-validation.

An earlier attempt trained a classifier on keyword-based labels. Too few comments carried a clear keyword, so it was abandoned. It remains in the notebook as a documented first attempt.

## How to Run

1. Open `emoji_analysis.ipynb` in Google Colab.
2. Upload `kaggle_RC_2019-05.csv` to the Colab session (left sidebar, Files).
3. Choose **Runtime > Run all**.
4. When the file picker appears near the end, select `cry_v2_labeled.csv` and `skull_v2_labeled.csv` together.

Checkpoints: the first code cell should print `File size (MB): 185.9` and `Rows loaded: 1000000`. After cleaning, the notebook should report 50 skull and 438 crying comments. If the validation cell reports that the similarity results look stale, rerun the similarity cell and then the validation cell. The first run downloads a small language model (about 90 MB). Generated CSVs are not downloaded automatically; set `DOWNLOAD_OUTPUTS = True` in the first code cell if you want copies.

## Limitations

- **Sample.** The skull sample is small (50 comments) and has no literal uses, so its literal rate is only bounded (roughly 0% to 7%). The data come from one platform and one month.
- **Labels.** Most labels were suggested by an LLM and reviewed by the author, not independently produced, and borderline cases (for example endearment and meme phrases such as "I'm not crying, you're crying") were decided by convention.
- **Embedding method.** The anchor phrases were chosen by hand, and the score measures closeness to them rather than literal use, so jokes about crying can score high. The model is a general-purpose English model. The 0.35 threshold was set before comparing with the labels; the threshold sweep is tuned on the same data it is evaluated on.
- **Classifier.** Only 42 crying comments were literal, so its scores are noisy.
- **Filtering.** Spam and non-English comments were removed with simple rules that may miss some cases.

## Tech Stack

Python, pandas, regular expressions, scikit-learn, sentence-transformers (`all-MiniLM-L6-v2`), matplotlib, seaborn, ipywidgets, Google Colab.

## Author

Ragavardhini Pattangi

## License

MIT License. See `LICENSE`.
