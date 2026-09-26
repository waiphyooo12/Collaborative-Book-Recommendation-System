# Book Recommendation System — Content-Based Filtering

A content-based book recommender built on user reading history and book synopses, implemented and compared using two different similarity approaches: **Cosine Similarity** (TF-IDF) and **Jaccard Similarity** (binary term vectors), with the second approach evaluated using Precision, Recall, and F-measure.

## Overview

Given a user's historical reading data (`UserData`, `UserHistoricalView`) and a book catalogue with synopses (`bookData`), the system:

1. Builds a **text profile per user** by concatenating the synopses of books they've already read.
2. Vectorizes book synopses and computes a **similarity matrix** between books.
3. For each user, scores their **unread books** by average similarity to books they've already read.
4. Returns the **top-N recommended books** per user.

## Approach 1 — TF-IDF + Cosine Similarity (`notebooks/part1_cosine_similarity.ipynb`)

- Vectorizes synopses with `TfidfVectorizer`
- Computes pairwise `cosine_similarity` between books
- Returns the **top 5** recommendations per user

## Approach 2 — Binary Vectors + Jaccard Similarity (`notebooks/part2_jaccard_similarity_evaluation.ipynb`)

- Vectorizes synopses with a binary `CountVectorizer`
- Computes pairwise `jaccard_score` between books
- Returns the **top 10** recommendations per user
- **Evaluates the recommender** against a held-out set of known-relevant books (`TestUserAnswers`), computing Precision, Recall, and F-measure per user

### Evaluation results (Part 2, n=9 test users)

| Metric    | Average |
| --------- | ------- |
| Precision | 0.027   |
| Recall    | 0.157   |
| F-measure | 0.043   |

## Repository structure

```
book-recommendation-system/
├── notebooks/
│   ├── part1_cosine_similarity.ipynb
│   └── part2_jaccard_similarity_evaluation.ipynb
├── data/
│   ├── part1_cosine_similarity/
│   │   ├── user_profiles.csv
│   │   ├── cosine_similarity_matrix.csv
│   │   └── recommendations.csv
│   └── part2_jaccard_similarity/
│       ├── user_profiles.csv
│       ├── jaccard_similarity_matrix.csv
│       ├── recommendations.csv
│       └── evaluation_metrics.csv
└── requirements.txt
```

## Tech stack

Python, pandas, NumPy, scikit-learn (`TfidfVectorizer`, `CountVectorizer`, `cosine_similarity`, `jaccard_score`)

## Running it

The notebooks expect the following raw inputs in the working directory (not included here — supply your own dataset with matching columns, or contact for the source data):

- `UserData.csv` — user records
- `UserHistoricalView.csv` — user reading history (userid, isbn)
- `bookData.csv` — book catalogue (isbn, Synopsis, ...)
- `TestUserAnswers.csv` — held-out relevant books per user (used for evaluation in Part 2)

Install dependencies and run the notebooks with Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Notes

- This was originally built as a group coursework project; the notebooks are shared here to showcase the recommendation-system methodology and evaluation approach.
- In the current evaluation code, Precision uses a fixed denominator (90) rather than each user's actual recommendation-list length — worth revisiting if you extend this project.
