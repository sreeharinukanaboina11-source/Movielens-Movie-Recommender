# Dataset: MovieLens Latest Small (ml-latest-small)

## Overview
This directory documents the dataset used in the CSE303 Machine Learning Week 10 Recommendation Systems Mini Project.
To maintain a clean Git repository, raw dataset files should not be committed to version control. Follow the instructions below to download and extract the official dataset.

## Source & Provenance
- **Dataset Provider:** GroupLens Research Group, Department of Computer Science and Engineering, University of Minnesota
- **Official Dataset Page:** [https://grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/)
- **Direct Download Link:** [https://files.grouplens.org/datasets/movielens/ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
- **Dataset Release Date:** September 26, 2018

## Academic Citation
If referencing this dataset in academic publications or coursework, cite:
> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

## Usage License & Terms
- GroupLens permits the use of this dataset for research and educational purposes.
- Commercial or revenue-bearing use requires explicit prior permission from the GroupLens Research Project at the University of Minnesota.
- Users may redistribute transformed versions provided they carry the same licensing terms and citations.

## Dataset Structure & Real Schemas
The downloaded archive `ml-latest-small.zip` unpacks into an `ml-latest-small/` directory containing four primary CSV files:

1. **`ratings.csv`** (100,836 rows, 4 columns):
   - `userId` (int64): Anonymized user identifier (610 unique users, each with >= 20 ratings).
   - `movieId` (int64): Unique movie identifier (9,724 unique movies observed in ratings).
   - `rating` (float64): Rating on a 5-star scale with 0.5 increments (0.5 to 5.0).
   - `timestamp` (int64): Epoch seconds in UTC.
2. **`movies.csv`** (9,742 rows, 3 columns):
   - `movieId` (int64): Unique movie identifier.
   - `title` (object): Movie title, usually including release year in parentheses.
   - `genres` (object): Pipe-separated genre list (e.g., `Adventure|Animation|Children|Comedy|Fantasy`).
3. **`tags.csv`** (3,683 rows, 4 columns):
   - `userId` (int64): User who applied the tag.
   - `movieId` (int64): Movie to which the tag was applied.
   - `tag` (object): Free-text descriptive tag.
   - `timestamp` (int64): Epoch seconds in UTC.
4. **`links.csv`** (9,742 rows, 3 columns):
   - `movieId` (int64): Internal MovieLens ID.
   - `imdbId` (int64): Identifier on IMDb.
   - `tmdbId` (float64): Identifier on The Movie Database (TMDB).

## Key Dataset Statistics
- **Total Observed Ratings:** 100,836
- **Unique Users:** 610
- **Unique Movies Rated:** 9,724 (out of 9,742 in movie catalog)
- **Catalog Movies with Zero Ratings:** 18 (used for cold-start item evaluation)
- **Duplicate User-Item Interactions:** 0
- **Matrix Sparsity:** 98.300% ($1 - \frac{100,836}{610 \times 9,724}$)

## Quick Download Instructions
To download and extract the dataset automatically using Python:
```bash
python -c "import requests, zipfile, io; r = requests.get('https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'); zipfile.ZipFile(io.BytesIO(r.content)).extractall('.')"
```
This extracts the `ml-latest-small` folder directly into the workspace root or project folder.
