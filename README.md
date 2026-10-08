# SMS Spam Checker (English only)

Demo app for the CP020003 final project: English SMS spam detection (TF-IDF + Logistic Regression / Naive Bayes / Linear SVM).

- Data: UCI SMS Spam Collection (Almeida, Gomez Hidalgo & Yamakami, 2011), CC BY 4.0
- The score is a sum of word weights, not a probability. Highlighted words are model weights, not proof that a message is a scam.
- English only. Thai or unrecognised text returns "cannot assess".
- Do not paste real private messages.

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`
