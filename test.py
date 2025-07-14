from main import PDFKnowledgeExtractor
import requests

# Initialize the extractor
extractor = PDFKnowledgeExtractor()

# # Example 1: Process a PDF file
# def process_pdf_example():
#     pdf_path = r"C:\Users\DELL\Desktop\Nishant Sir\khaitan_chatbot\Alumni Newsletter-KConnect APRIL 2025.pdf"  # Replace with actual PDF path
#     result = extractor.process_pdf(pdf_path)
#     print("PDF Processing Result:", result)

# # Example 2: Query the knowledge base
# def query_example():
#     questions = [
#         "What is the firm about?",
#         "Where are the offices located?",
#         "What programs does the firm have?",
#         "What technology tools are used?",
#         "What awards has the firm received?"
#     ]
    
    # for question in questions:
    #     result = extractor.find_similar_question(question)
    #     print(f"\nQuestion: {question}")
    #     if result:
    #         print(f"Answer: {result['answer']}")
    #     else:
    #         print("No relevant answer found.")

# # Example 3: Direct text processing (without API)
# def direct_processing_example():
#     # Extract text from PDF
#     pdf_path = r"C:\Users\DELL\Desktop\Nishant Sir\khaitan_chatbot\Alumni Newsletter-KConnect APRIL 2025.pdf"
#     text = extractor.extract_text_from_pdf(pdf_path)
#     print("Extracted text length:", len(text))
    
#     # Save text to file
#     extractor.save_text_to_file(text, pdf_path)
#     print("Text saved to:", extractor.txt_file_path)
    
#     # Categorize the text
#     categories = extractor.categorize_text(text)
#     print("Categories found:", list(categories['categories'].keys()))
#     print("Q&A pairs generated:", len(categories['qa_pairs']))

# Example 4: API client example (for testing the API endpoints)

def test_api_endpoints():
    base_url = "http://localhost:8000"
    
    # Test health check
    response = requests.get(f"{base_url}/health")
    print("Health check:", response.json())
    
    # Test PDF processing
    pdf_data = {"pdf_path": r"C:\Users\DELL\Desktop\Nishant Sir\khaitan_chatbot\Alumni Newsletter-KConnect APRIL 2025.pdf"}
    response = requests.post(f"{base_url}/process-pdf", json=pdf_data)
    print("PDF processing:", response.json())
    
    # Test query
    query_data = {"question": "What is the firm about?"}
    response = requests.post(f"{base_url}/query", json=query_data)
    print("Query result:", response.json())
    
    # Get knowledge base
    response = requests.get(f"{base_url}/knowledge-base")
    print("Knowledge base keys:", list(response.json().keys()))

if __name__ == "__main__":
    # Run examples
    print("=== PDF Processing Example ===")
    # process_pdf_example()  # Uncomment and provide PDF path
    
    print("\n=== Query Example ===")
    # query_example()  # Uncomment after processing a PDF
    
    print("\n=== Direct Processing Example ===")
    # direct_processing_example()  # Uncomment and provide PDF path
    
    print("\n=== API Testing Example ===")
    # test_api_endpoints()  # Uncomment after starting the API server