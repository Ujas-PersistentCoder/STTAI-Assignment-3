# **Multimodal Sentiment Engine: Data Augmentation & Validation Pipeline**

## **Project Overview**

This project implements a comprehensive NLP and Speech processing pipeline designed to expand a small, high-quality sentiment dataset into a robust, multimodal, and multilingual training resource. Starting with only 300 labelled samples, the engine uses classical augmentation, LLM-based synthetic generation, and multilingual back-translation to scale the dataset to more than twice its size. Quality is validated at every step using Speech-to-Text (Whisper), Self-BLEU diversity metrics, and Deep Learning evaluators.

## **Key Features**

* **Data Consolidation:** Merges gold-standard, weak, and LLM labels using a high-confidence Logistic Regression filter (Confidence ![][image1]).  
* **Classical Augmentation:** Implements WordNet-based synonym replacement and Hindi back-translation to balance minority classes.  
* **LLM Synthetic Generation:** Uses OpenRouter API with few-shot prompting to generate 300+ diverse reviews, verified for consistency.  
* **Multilingual Support:** Translates English reviews to Hindi with sentiment preservation checks.  
* **Multimodal Testing:** Generates synthetic audio reviews (gTTS) and validates them using OpenAI Whisper (WER) with a custom normalization pipeline.  
* **Black-Box Evaluation:** Measures accuracy gains using a frozen PyTorch text embedder against a human-annotated gold standard.

## **Repository Structure**

* `consolidated\_base.csv`: The initial filtered and merged dataset (Task 1).  
* `augmented\_classical.csv`: Minority class samples expanded via synonyms and back-translation (Task 1).  
* `llm\_generated\_300.csv`: 300 synthetic reviews generated via LLM (Task 2).  
* `llm\_generated\_flagged.csv`: Reviews where LLM labels and baseline model predictions mismatched (Task 2).  
* `diversity\_report.txt`: Self-BLEU scores tracking the variety of synthetic data (Task 2).  
* `prompt\_template.txt`: The few-shot prompt used for LLM generation (Task 2).  
* `bilingual\_reviews.csv`: Hindi-translated reviews with BLEU and quality flags (Task 3).  
* `audio\_samples/`: Directory containing 30 generated .wav review files (Task 4).  
* `audio\_features.csv`: Spectral features (MFCCs, Spectral Centroid) extracted via librosa (Task 4).  
* `audio\_validation.csv`: Whisper transcription results and Word Error Rate (WER) scores (Task 4).  
* `final\_augmented\_dataset.csv`: The final deduplicated and cleaned training set (Task 5).  
* `evaluator.py`: The Python script containing the BlackBoxEvaluator class.  
* `text\_embedder.pt`: The frozen PyTorch model used for deep feature extraction.

## **Setup & Installation**

### **1\. Prerequisites**

* **Python 3.8+**  
* **FFmpeg:** Required for OpenAI Whisper and audio processing.  
  * *Windows:* Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add the bin folder to your System PATH.  
  * *Verification:* Run ffmpeg \-version in your terminal.

### **2\. Model Download**

The evaluator requires a frozen PyTorch model. Download it here and place it in the root directory:

* [**Download text\_embedder.pt**](https://drive.google.com/file/d/1Yu7YDA4a2CDVUmVjH9vWUIT2aJCfyBKN/view?usp=sharing)

### **3\. Install Dependencies**

`pip install \-r requirements.txt`

### **4\. API Configuration**

Create a .env file in the root directory to store your OpenRouter credentials:

`OPENROUTER\_API\_KEY=your\_sk\_or\_key\_here`

## **Pipeline Walkthrough**

### **Task 1: Consolidation & Classical Augmentation**

We merge gold\_standard\_100.csv, weak\_labels\_200.csv, and a high-confidence subset of llm\_labels\_150.csv. To balance classes, we apply Synonym Replacement (WordNet) and Back-Translation to the minority class, filtering results via Jaccard Similarity (Threshold: ![][image2]).

### **Task 2: LLM Synthetic Generation**

Using OpenRouter (GPT-3.5/4), we generate 300 reviews. Diversity is measured via **Self-BLEU** (Target ![][image3]). We flag mismatches where our baseline Logistic Regression model disagrees with the LLM-provided label.

### **Task 3: Multilingual Translation**

100 samples are translated to Hindi and back to English. We validate quality using:

* **BLEU Score:** ![][image4].  
* **Sentiment Preservation:** Ensuring the model's prediction remains constant after translation.

### **Task 4: Multimodal Audio Pipeline**

We convert text reviews to speech:

* **TTS:** gTTS (American English) converted to .wav.  
* **Feature Extraction:** 13 MFCC means, Spectral Centroid, and Zero Crossing Rate.  
* **Whisper Validation:** Local transcription via whisper-tiny. We implement a custom WER function that normalizes text (removing punctuation/contractions) to ensure fair evaluation. Samples with **WER** ![][image5] are flagged.

### **Task 5: Performance Evaluation**

The BlackBoxEvaluator uses deep linguistic embeddings to compare performance between the baseline model (trained on the consolidated data) and the augmented model (trained on the full final dataset).

## **Final Summary**

The effectiveness of this pipeline is demonstrated by the performance uplift measured against the human-annotated gold standard. By enriching a small dataset with synthetically generated and multimodal data, we achieve a more generalised sentiment engine capable of handling varied inputs across different languages and modalities.

### **Key Insights**

* **Linguistic Diversity:** LLM-generated samples provided the biggest boost in semantic coverage, while classical methods effectively mitigated class imbalance.  
* **Robustness:** Multilingual back-translation served as an effective paraphrasing tool, enhancing the model's resistance to minor phrasing changes.  
* **Multimodal Validation:** The Whisper-based validation ensures that the pipeline is ready for future voice-integrated sentiment features.

## **Acknowledgements**

This was developed as **Assignment 3** for the **STTAI (Software Tools and Techniques for AI)** course at **IIT Gandhinagar**. I would like to acknowledge that several core components and datasets utilised in this pipeline were provided as instructor-led assets and are not original work:

* **Base Datasets:** gold\_standard\_100.csv, weak\_labels\_200.csv, and llm\_labels\_150.csv.  
* **Boilerplate Code:** Initial logic and structure provided in the starter.py / starter.ipynb template.  
* **Evaluation Assets:** The evaluator.py script and the pre-trained text\_embedder.pt deep learning model.

The primary focus of this work was the implementation of the augmentation logic, multilingual translation validation, multimodal feature extraction, and the synthesis of these components into a unified sentiment engine.
