import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import csv

#pip install langchain_openai
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#uncomment these on first run through to install the nlp libraries
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

###cosine code
#Preprossessing
def preprocess(text):
    #tokenizing the text
    words = word_tokenize(text)

    #next is removing stopwords and any punctuation
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    #stemming the words 
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in filtered_words]

    #join tokens back to string for vectorization
    processed_text = ' '.join(stemmed_words)
    return processed_text

def cosine_sim(text1, text2):

    processed_paragraph1 = preprocess(text1)
    processed_paragraph2 = preprocess(text2)

    # Using TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform([processed_paragraph1, processed_paragraph2])
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=['Paragraph1', 'Paragraph2'])

    #calculating the cosine similarity between the feature representations
    cosine_sim = cosine_similarity([df_tfidf.loc['Paragraph1']], [df_tfidf.loc['Paragraph2']])

    return cosine_sim[0][0]


import os
import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)



###Jaccard code
def preprocess_text2(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

    # Stemming
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in filtered_words]

    return stemmed_words




def jaccard(text1, text2):
    #Run the preprocessing
    text1_words = preprocess_text2(text1)
    text2_words = preprocess_text2(text2)

    #Code to calculate the Jaccard similarity of the two sets
    intersection = len(set(text1_words) & set(text2_words))
    union = len(set(text1_words) | set(text2_words))
    jaccard_similarity = intersection / union if union != 0 else 0

    return jaccard_similarity









#Simple gen
def generate(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]  
    )
    content = response.choices[0].message.content    
    return content

#Rag Gen
#RAG Algorithm
api_key=os.environ.get('OPENAI_API_KEY')

def rag_gen(context, question):

    #parser is used to change the output into a string
    parser = StrOutputParser()

    #defining the model
    model = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")

    #the template for the prompt being passed to the LLM
    template = """
    Answer the question based on the context below.

    Context: {context}

    Question: {question}
    """

    #defining the prompt with the template
    prompt = ChatPromptTemplate.from_template(template)

    #the chain that the model follows with the response generation
    chain = prompt | model | parser

    #this code generates the response
    #using the passed context and question
    response = chain.invoke({
    "context": context,
    "question": question
    })

    return response


def ai_ditect(sustext, method):

    data = []
    promptsim = []
    if method == "jaccard":
        print("Runing Jaccard Similarity")
        for x in numberofresponses:
            response_text = []
            for y in range(x):
                response_text.append(generate(prompt))

            for z in range(len(response_text)):
                promptsim.append(jaccard(sustext,response_text[z]))
            print(promptsim)

            data.append(max(promptsim))
    elif method == "cosine":
        print("Runing Cosine Similarity")
        for x in numberofresponses:
            response_text = []
            for y in range(x):
                response_text.append(generate(prompt))

            for z in range(len(response_text)):
                promptsim.append(jaccard(sustext,response_text[z]))
            print(promptsim)

            data.append(max(promptsim))

    print(data)

    file_path = "normal.csv"

    # Open the file in write mode with csv writer
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write each element of the array as a row in the CSV file
        writer.writerow(data)
    
    return data


def ai_ditect_rag(sustext, method):

    data = []
    promptsim = []
    if method == "jaccard":
        print("Runing Jaccard Similarity")
        for x in numberofresponses:
            response_text = []
            for y in range(x):
                response_text.append(rag_gen(sustext,prompt))

            for z in range(len(response_text)):
                promptsim.append(jaccard(sustext,response_text[z]))
            print(promptsim)

            data.append(max(promptsim))
    elif method == "cosine":
        print("Runing Cosine Similarity")
        for x in numberofresponses:
            response_text = []
            for y in range(x):
                response_text.append(rag_gen(sustext,prompt))

            for z in range(len(response_text)):
                promptsim.append(jaccard(sustext,response_text[z]))
            print(promptsim)

            data.append(max(promptsim))

    print(data)

    file_path = "rag.csv"

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(data)
    
    return data


# Example prompt
prompt = "Write a paragraph about dogs"
humantext = "Dogs are animals that can come in many different forms. The ancestor of the dog is the wolf, and a dog is derived from the domestication of the wolf. The grey wolf is the closest living relative to the dog. The average lifespan of dogs ranges per breed. Dogs are one of the most popular pets in the world. Some specific breeds of dogs are controversial and have been banned throughout the United Kingdom. This is because some breeds of dogs are more aggressive than others or even just have had a poor upbringing. This is not true for all types of dogs, as some dogs contain the capacity to assist humankind in the form of guide dogs and work dogs."
aitext = "Dogs are often referred to as man's best friend, and for good reason. They are loyal, loving, and incredibly social animals that form strong bonds with their owners. Dogs come in all shapes, sizes, and breeds, each with their own unique personality traits and characteristics. From small lap dogs to large working breeds, there is a dog out there for everyone. Dogs are known for their playful nature, their ability to protect and serve, and their unwavering loyalty to their human companions. They are not only great pets but also wonderful companions that bring joy and happiness to their owners' lives. Whether they are running around in the yard, cuddled up on the couch, or out for a walk, dogs never fail to bring a smile to their owner's face."
#numberofresponses = [1,4,5,10,20]
numberofresponses = [20]

method = "cosine"

#ai_ditect(humantext, method)
#ai_ditect(aitext, method)

#some rag tests
#ai_ditect_rag(humantext, method)
#ai_ditect_rag(aitext, method)


prompt = "Write a paragraph about what discovery in the last 100 years has been most beneficial"
humantext = "Computers are among the most beneficial discoveries of the past 100 years. They have enabled people to communicate worldwide and stay connected to their friends and family. The sheer convenience of computing simple algorithms on even handheld devices has enabled the betterment of society. They have also enabled the average person to access seemingly infinite resources of information. They have helped society through the vast amounts of data that they can store and process and made tasks that would require alot of time to complete process faster."
aitext = "The discovery of computers has revolutionized the way we live our lives. From facilitating communication to streamlining business operations and even helping in scientific research, computers have become an integral part of our daily activities. They have made tasks that were once time-consuming and laborious much more efficient and streamlined, allowing us to accomplish more in less time. Computers have also opened up new opportunities for learning and development, providing access to a wealth of information and resources at our fingertips. Overall, the discovery of computers has significantly improved the way we work, communicate, and interact with the world around us."

#rag vs normal for opinionated prompt
print("NORMAL " , ai_ditect(aitext, method))
print("Ai RAG: " , ai_ditect_rag(aitext, method))
print("Human RAG: " , ai_ditect_rag(humantext, method))


