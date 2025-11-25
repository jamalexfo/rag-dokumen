import sys
from rag_core import initialize_rag_chain

def main():
    try:
        rag_chain = initialize_rag_chain()
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        sys.exit(1)

    print("RAG System Ready! (Type 'exit' to quit)")
    print("-" * 50)

    # Interactive Loop
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query.strip():
            continue

        try:
            response = rag_chain.invoke({"query": query})
            answer = response.get('result', '')
            
            # Check if answer is NOT_FOUND, fallback to base model
            if "NOT_FOUND" in answer:
                print("\nAnswer not found in context, using base model...")
                from langchain_google_genai import GoogleGenerativeAI
                import os
                
                base_llm = GoogleGenerativeAI(
                    model="models/gemini-2.5-flash",
                    temperature=0.7,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                answer = base_llm.invoke(query)
                print(f"\nAnswer (Base Model): {answer}")
            else:
                print(f"\nAnswer: {answer}")
                
                # Show sources
                if 'source_documents' in response:
                    print("\nSources:")
                    for doc in response['source_documents']:
                        print(f"- {doc.metadata.get('source', 'Unknown')}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
