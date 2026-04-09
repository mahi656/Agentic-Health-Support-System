import os
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def health_agent_response(query, patient_data, risk_prob, vectorstore=None):
    """
    RAG-powered LLM consulting agent using Groq's Llama 3.1.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "**Error:** `GROQ_API_KEY` is not set. Please add it to your `.env` file to activate the AI Agent."
        
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    patient_context = f"""
    Patient Profile:
    - Name: {patient_data.get('name', 'Unknown')}
    - Age: {patient_data.get('age', 'Unknown')}
    - Gender: {'Male' if patient_data.get('sex', 0) == 1.0 else 'Female'}
    - BMI: {patient_data.get('bmi', 'Unknown')}
    
    Clinical Vitals:
    - Blood Pressure: {patient_data.get('trestbps', 'Unknown')} mmHg
    - Cholesterol: {patient_data.get('chol', 'Unknown')} mg/dL
    - Max Heart Rate: {patient_data.get('thalach', 'Unknown')} BPM
    - Fasting Blood Sugar: {'> 120' if patient_data.get('fbs', 0) == 1.0 else '< 120'} mg/dL
    
    Current Model Assessment:
    - Heart Disease Risk Probability: {risk_prob * 100:.1f}%
    """

    system_prompt = (
        "You are MediRisk AI, an intelligent clinical assistant.\n"
        "You have access to the patient's current clinical profile and assessment metrics.\n"
        "If context documents are provided below, use them to provide medical guidelines, history summaries, or best practices.\n"
        "If the query is unrelated to health or the patient's profile, politely redirect the conversation to health matters.\n\n"
        "PATIENT CONTEXT:\n{patient_context}\n\n"
        "RETRIEVED DOCUMENTS (if any):\n{context}\n\n"
        "Answer the user's query professionally, accurately, and compassionately. Try to format your output using markdown for readability."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({
            "input": query,
            "patient_context": patient_context
        })
        return response["answer"]
    else:
        chain = prompt | llm
        response = chain.invoke({
            "input": query,
            "patient_context": patient_context,
            "context": "No external documents provided."
        })
        return response.content