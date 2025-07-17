# -*- coding: utf-8 -*-
"""RAG_bases_Question_Answering_System.ipynb


#importing necessary libraries and installations
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

!pip install --upgrade langchain langchain-community ctransformers
!pip install langchain
!pip install ctransformers
!pip install ctransformers[cuda]
!pip install langchain-community
!pip install -U sentence-transformers

import os
from langchain_community.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from ctransformers import AutoModelForCausalLM

#hugging face login
!huggingface-cli login

#mounting google drive 
from google.colab import drive
drive.mount('/content/drive')


#RAG Class
class RAG :
      def __init__(self) :
            # Embedding model initialization
            from sentence_transformers import SentenceTransformer
            self.model_1_sentence_bert =  SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            # Mistral_model_initialization
            os.environ['XDG_CACHE_HOME'] = 'drive/MyDrive/LLM_data/model_mistral/cache/'
            self.model_2_mistral_LLM =  AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=50)

            # Embedding using SentenceBERT and return data_list
            self.data = pd.read_csv("/content/drive/MyDrive/LLM_data/data_wansuk_vangdu.csv")
            self.data_list = list(self.data['sentences'])
            self.embedding = (self.model_1_sentence_bert.encode(self.data_list))

            return None


      #user query embedding
      def perform_query_embedding(self, query) :
            # query_embedding
            query_embedding = self.model_1_sentence_bert.encode(query)
            print(query)

            return query_embedding



       #retrieval function
      def finding_most_similar_chunk(self, query_embedding):
          # finding most similar chunk and retrieving it
            max_similarity = -1
            most_similar_chunk = None
            for i in range(len(self.embedding)):
                similarity = 1 - cosine(self.embedding[i], query_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_chunk = self.data_list[i]
                    print(most_similar_chunk)
                    print(max_similarity)

            return max_similarity, most_similar_chunk


            
      #generation function
      def generate_answer(self,query):
            query_embedding = self.perform_query_embedding(query)
            most_similar_chunk = self.finding_most_similar_chunk(query_embedding)
            self.responce = self.model_2_mistral_LLM(f"[INST] You are an agent who answers questions about preferences of a person. Based on the information provided below about a personality in INFO, you need to answer question given in QUESTION . Restrict your knowledge by using only the information provided below. INFO: {most_similar_chunk}   Dont go beyond this information provided. QUESTION : {query} [/INST]")
            print(self.responce)



#object initialisation 

query_question_1 = "Where wansuk vangdu going in August"
query_question_2 = "Where wansuk vangdu going in September"
query_question_3 = "Where In the USA Wansuk Vangdu wants to visit"
RAG_object = RAG()

answer_1 = RAG_object.generate_answer(query_question_1)
answer_2 = RAG_object.generate_answer(query_question_2)
answer_3 = RAG_object.generate_answer(query_question_3)

