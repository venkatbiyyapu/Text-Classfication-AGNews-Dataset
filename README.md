# AGNews Text Classification with Spark NLP

This project implements a text classification model using Spark NLP to classify the AGNews dataset. The goal is to compare the performance of different pipelines using BERT and RoBERTa embeddings, with and without text preprocessing steps such as lemmatization and stop word removal.

## Dataset

The AGNews dataset contains news articles categorized into four classes: 
- World
- Sports
- Business
- Sci/Tech

## Steps Implemented

### a) BERT Embeddings without Preprocessing

I used **BERT embeddings** (`bert_base_cased`) and a generic annotator model called `ClassifierDL` in Spark NLP to train a model without any text preprocessing. The model was trained for 5 epochs, and the test accuracy was recorded.

- **Pipeline**: Raw text → BERT embeddings → ClassifierDL
- **Test Accuracy**: **0.8723**

### b) Preprocessing: Lemmatization and Stop Word Removal

I added preprocessing steps to the pipeline and evaluated the impact of each step (individually and combined) on the model's performance.

1. **Lemmatization only**:
   - **Pipeline**: Raw text → Lemmatization → BERT embeddings → ClassifierDL
   - **Test Accuracy**: **0.8593**

2. **Stop Word Removal only**:
   - **Pipeline**: Raw text → Stop Word Removal → BERT embeddings → ClassifierDL
   - **Test Accuracy**: **0.8621**

3. **Lemmatization + Stop Word Removal**:
   - **Pipeline**: Raw text → Lemmatization → Stop Word Removal → BERT embeddings → ClassifierDL
   - **Test Accuracy**: **0.8581**

### Best Performing Pipeline

The highest test accuracy for the BERT base cased model was observed without preprocessing (**0.8723**), emphasizing the significance of its contextual embeddings. Preprocessing steps such as lemmatization and stop word removal, which are standard, actually resulted in minor reductions in accuracy (0.8593 and 0.8621, respectively). 

This suggests that:
- **Lemmatization**: May remove context crucial for BERT's contextualized understanding.
- **Stop Word Removal**: Could impede the model's comprehension of sentence structure and semantics.

The combination of both preprocessing steps yielded slightly lower accuracy (0.8581), indicating that their joint application did not positively impact the model's performance. The results emphasize that BERT's inherent contextualization reduces the need for traditional preprocessing, and preprocessing decisions must be task-specific to avoid trade-offs in performance.

### c) RoBERTa Embeddings

Since the `bert_base_cased` model achieved the highest accuracy without preprocessing, I replaced the BERT embeddings with **RoBERTa embeddings** (`roberta_base`) and retrained the model without preprocessing.

- **Pipeline**: Raw text → RoBERTa embeddings → ClassifierDL
- **Test Accuracy**: **0.7986**

### Final Comparison and Explanation

- **BERT embeddings (`bert_base_cased`)**: **0.8723**
- **RoBERTa embeddings (`roberta_base`)**: **0.7986**

Although BERT and RoBERTa share similar transformer architectures, their divergence in training objectives can affect performance on specific tasks. RoBERTa uses dynamic masking and does not include a next-sentence prediction task, while incorporating more training data. These differences can lead to variations in generalization to downstream tasks.

In this case, the pretraining data and task alignment favored BERT, which achieved better accuracy than RoBERTa. RoBERTa's decline in performance may suggest that its pretraining data did not align as well with the downstream AGNews task. To improve RoBERTa’s performance, further experimentation with hyperparameter tuning, preprocessing strategies, or trying other transformer variants could be beneficial.

## Conclusion

This project highlights the trade-offs in text preprocessing when using contextual embeddings like BERT and RoBERTa. The findings emphasize that preprocessing decisions should be based on the specific task at hand, as traditional methods may not always improve performance with modern embeddings.
