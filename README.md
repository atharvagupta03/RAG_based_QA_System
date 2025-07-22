 RAG-Based Question Answering System

This project is a **Retrieval-Augmented Generation (RAG)** based Question Answering system that uses Sentence-BERT embeddings and the **Mistral 7B Instruct** language model (quantized with GGUF via CTransformers) to answer questions from a structured dataset. The system restricts its responses to the content of the input data, avoiding hallucination or external knowledge.

---

##  Objective

To create a lightweight, local Q&A system that:
- Retrieves the most relevant sentence from a personal dataset
- Answers user queries based only on the retrieved sentence using a powerful LLM

---

##  Tech Stack

- **Python**
- **Pandas / NumPy**
- **Sentence-BERT** (`all-MiniLM-L6-v2`)
- **CTransformers** (for running LLMs locally)
- **LangChain (Community)**
- **Mistral-7B-Instruct GGUF model**
- **Google Colab + Google Drive** for environment and data handling

---

##  How It Works

### 1. **Load and Encode Dataset**
- Loads a CSV (`wiki_page_content_on_fraud_websites.csv`) with a column `page_content`
- Embeds all sentences using Sentence-BERT

### 2. **Process User Query**
- Converts the user question into an embedding using the same model

### 3. **Retrieve Relevant Context**
- Computes cosine similarity between the query and all sentence embeddings
- Selects the most similar sentence from the dataset

### 4. **Generate Answer**
- Constructs a prompt using:
  - Retrieved sentence as `INFO`
  - User's query as `QUESTION`
- Sends the prompt to **Mistral 7B** via CTransformers for response generation

---

##  File Structure
├── RAG_bases_Question_Answering_System.ipynb # Main notebook

├── data/

│ └── wiki_page_content_on_fraud_websites.csv # CSV containing sentences

├── drive/LLM_data/model_mistral/ # Folder for GGUF model files


##  Setup & Usage Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/rag-question-answering.git
cd rag-question-answering
``` 
### Step 2: Open in Google Colab
Upload the notebook: RAG_bases_Question_Answering_System.ipynb

Open it in Google Colab
Mount Google Drive:
from google.colab import drive
drive.mount('/content/drive')


### Step 3: Install Dependencies

- !pip install --upgrade langchain langchain-community ctransformers
- !pip install langchain
- !pip install ctransformers
- !pip install ctransformers[cuda]
- !pip install langchain-community
- !pip install -U sentence-transformers
- !huggingface-cli login

Make sure to have necessary data files. You can change the dataset and use the RAG System according to your needs. 

