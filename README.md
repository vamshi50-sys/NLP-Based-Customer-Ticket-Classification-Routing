\# AI Customer Support Ticket Classification and Routing using NLP



An NLP-based application that classifies customer support tickets, assigns priorities, and routes tickets to the appropriate support team. Low-confidence predictions are sent to a human review queue instead of being automatically routed.



\## Project Overview



Customer support teams receive tickets covering different issues, such as order cancellations, refunds, delivery problems, and account access.



This project uses Natural Language Processing (NLP) and Machine Learning to identify the intent of a customer's message and support the ticket-routing process.



\## Features



\* \*\*Intent Classification:\*\* Predicts the customer's intent across 27 classes.

\* \*\*Text Processing:\*\* Applies text normalization and word-level and character-level TF-IDF feature extraction.

\* \*\*Machine Learning:\*\* Uses Logistic Regression for intent classification.

\* \*\*Priority Assignment:\*\* Assigns HIGH, MEDIUM, or LOW priority using rule-based business logic.

\* \*\*Team Routing:\*\* Maps predicted intents to support teams.

\* \*\*Confidence-Based Review:\*\* Sends predictions below a configurable threshold to a human review queue.

\* \*\*Ticket Dashboard:\*\* Displays ticket statistics and analysis results.

\* \*\*Ticket Queue:\*\* Supports searching and filtering tickets by priority, decision, and support team.

\* \*\*Gradio Interface:\*\* Provides an interactive web application.



\## Model Performance



The combined word-level and character-level TF-IDF Logistic Regression model achieved \*\*99.78% accuracy\*\* on the held-out test set.



| Model                                                  | Test Accuracy |

| ------------------------------------------------------ | ------------: |

| Word-level TF-IDF + Logistic Regression                |        99.07% |

| Character-level TF-IDF + Logistic Regression           |        99.76% |

| Combined Word + Character TF-IDF + Logistic Regression |        99.78% |



These results are based on a train/test split of the selected dataset. Performance on real-world customer messages may differ.



\## Technology Stack



\* Python

\* Pandas

\* Scikit-learn

\* TF-IDF

\* Logistic Regression

\* SciPy

\* Joblib

\* Gradio



\## Dataset



\*\*Bitext Customer Support Dataset\*\*



Source: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset



The dataset contains customer-support messages, categories, intents, and responses. The `instruction` field is used as the input text, and `intent` is the classification target.



\## Project Structure



```text

AI-Customer-Support-Ticket-Classifier/

├── models/

│   ├── word\_tfidf.joblib

│   ├── char\_tfidf.joblib

│   └── combined\_model.joblib

├── app.py

├── NLP.ipynb

├── requirements.txt

└── README.md

```



\## Run Locally



1\. Clone or download this repository.



2\. Open a terminal in the project directory.



3\. Install the required packages:



&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```



4\. Start the application:



&#x20;  ```bash

&#x20;  python app.py

&#x20;  ```



5\. Open the local URL displayed in the terminal.



\## How Ticket Routing Works



1\. The customer enters a support message.

2\. The text is normalized and converted into TF-IDF features.

3\. The trained model predicts the ticket intent.

4\. Business rules assign a priority and support team.

5\. Predictions meeting the confidence threshold are automatically routed.

6\. Lower-confidence predictions are sent to the Human Review Queue.



\## Important Notes



\* Priority assignment is rule-based; it is not a separately trained machine-learning model.

\* The confidence threshold is configurable. Logistic Regression's predicted probabilities are not necessarily calibrated confidence estimates.

\* Support-team assignments are defined by the project's routing rules.

\* This is a prototype intended for demonstration and learning, not a production ticketing system.



\## Future Improvements



\* Evaluate the system on real-world customer-support messages.

\* Improve handling of ambiguous and out-of-domain tickets.

\* Calibrate prediction probabilities.

\* Add persistent ticket storage.

\* Explore transformer-based text classification.



