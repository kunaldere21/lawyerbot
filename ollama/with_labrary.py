import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Load the Ollama model (LLaMA)
def load_llama():
    llm = OllamaLLM(model="llama2")  # Available models: "llama2", "mistral", etc.
    return llm

# Define a custom prompt template
CUSTOME_PROMPT_TEMPLATE = """
You are a helpful and accurate AI assistant.

Question: {question}
Answer:
"""

# Create the chain
def create_chain(llm):
    prompt = PromptTemplate.from_template(CUSTOME_PROMPT_TEMPLATE)
    chain = prompt | llm
    return chain

if __name__ == "__main__":
    # Load the model
    llm = load_llama()
    
    # Create the LLM Chain
    chain = create_chain(llm)
    
    # Test with a user query
    USER_QUERY = input("Ask something: ")
    response = chain.invoke({"question": USER_QUERY})
    print("Response:", response)
