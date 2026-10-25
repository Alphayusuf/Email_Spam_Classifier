# Email Classification Research Lab

This is an upgraded Streamlit prototype for an MSc project comparing Multinomial Naive Bayes and Linear Support Vector Machine (SVM).

## Features
- Research dashboard and class distribution
- Dataset exploration (lengths, duplicate count, sample preview)
- Controlled TF-IDF and model settings
- Held-out stratified test evaluation
- Accuracy, spam precision/recall/F1, classification reports and confusion matrices
- ROC and precision-recall curves
- Misclassification and model disagreement analysis
- Single-message predictions from both models
- CSV downloads for metrics, predictions, reports and experiment settings

## Expected dataset format
`emails.csv` must contain:
- `text`: email text
- `spam`: `0` for legitimate/ham, `1` for spam

## Run locally
1. Install Python 3.10 or later.
2. Extract this ZIP.
3. Open a terminal in the extracted folder.
4. Install packages:
   `python -m pip install -r requirements.txt`
5. Run:
   `python -m streamlit run app.py`

## Deploy on Streamlit Community Cloud
Upload `app.py`, `emails.csv`, and `requirements.txt` to the connected GitHub repository, then redeploy/reboot the Streamlit app.

## Academic integrity and evaluation notes
- The application evaluates models on a held-out test set. Do not tune settings repeatedly against that test set and then report it as an unbiased final estimate.
- For a stronger final experiment, use cross-validation on the training set to select settings, then run the final evaluation once on the held-out test set.
- Confirm the original source and licensing of the included `emails.csv` before naming it as the official SpamAssassin corpus in your dissertation.
- Do not claim these results are better than Gmail's system. No direct comparison with Gmail is performed.
- Predictions are not guarantees that a message is safe or malicious.
- Avoid using confidential or personally identifiable email messages.
