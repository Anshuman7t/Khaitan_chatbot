import os
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import spacy


class PDFKnowledgeExtractor:
    def __init__(self, txt_file_path: str = "knowledge_base.txt", 
                json_file_path: str = "knowledge_base.json"):
        self.txt_file_path = txt_file_path
        self.json_file_path = json_file_path
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully")

        print("Loading spaCy model for sentence splitting...")
        self.spacy_nlp = spacy.load("en_core_web_sm")
        print("spaCy loaded.")

        print("Loading T5 question generator...")
        self.qg_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl")
        self.qg_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-hl")
        print("T5 question generator loaded.")
        
        print("Loading knowledge base...")
        self.knowledge_base = self.load_knowledge_base()
        print(f"Knowledge base loaded with {len(self.knowledge_base.get('qa_pairs', []))} Q&A pairs")
        
        # Rest of the greeting responses remain the same...
        self.greeting_responses = {
            "hi": "Hello! I'm here to help you with questions about the document. What would you like to know?",
            "hello": "Hi there! I can answer questions about the document content. How can I assist you?",
            "hey": "Hey! I'm your document assistant. Feel free to ask me anything about the content.",
            "good morning": "Good morning! I'm ready to help you with any questions about the document.",
            "good afternoon": "Good afternoon! How can I help you with the document today?",
            "good evening": "Good evening! I'm here to assist you with document-related questions.",
            "thanks": "You're welcome! Is there anything else you'd like to know about the document?",
            "thank you": "You're very welcome! Feel free to ask if you have more questions.",
            "bye": "Goodbye! Feel free to come back if you have more questions about the document.",
            "goodbye": "Goodbye! Have a great day!",
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def save_text_to_file(self, text: str, pdf_path: str):
        """Save extracted text to txt file with metadata"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n{'='*50}\n"
        header += f"Document: {os.path.basename(pdf_path)}\n"
        header += f"Extracted on: {timestamp}\n"
        header += f"{'='*50}\n\n"
        
        # Append to existing file or create new
        mode = 'a' if os.path.exists(self.txt_file_path) else 'w'
        with open(self.txt_file_path, mode, encoding='utf-8') as f:
            f.write(header + text + "\n\n")
    
    def is_greeting_or_casual(self, text: str) -> Optional[str]:
        """Check if the input is a greeting or casual conversation"""
        text_lower = text.lower().strip()
        
        # Check for exact matches
        if text_lower in self.greeting_responses:
            return self.greeting_responses[text_lower]
        
        # Check for partial matches
        greeting_keywords = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
        for keyword in greeting_keywords:
            if keyword in text_lower and len(text_lower) <= len(keyword) + 5:
                return self.greeting_responses.get(keyword, "Hello! How can I help you with the document?")
        
        # Check for very short inputs that might be casual
        if len(text_lower) <= 3 and not any(char.isalpha() for char in text_lower):
            return "I'm here to help you with questions about the document. What would you like to know?"
        
        return None
    
    def categorize_text(self, text: str) -> Dict[str, Any]:
        """Categorize text into structured knowledge base"""
        categories = {
            "firm_overview": self.extract_firm_overview(text),
            "offices_locations": self.extract_offices(text),
            "programs_initiatives": self.extract_programs(text),
            "technology_tools": self.extract_technology(text),
            "awards_recognition": self.extract_awards(text),
            "practice_areas": self.extract_practice_areas(text),
            "general_info": self.extract_general_info(text),
            "people_contacts": self.extract_people_contacts(text),
            "events_news": self.extract_events_news(text)
        }
        
        # Generate comprehensive Q&A pairs
        qa_pairs = self.generate_comprehensive_qa_pairs(categories, text)

        # Also add automatically generated Q&A from full text using T5
        auto_qna = self.generate_qna_from_sentences_with_t5(text, max_qna=100)
        qa_pairs.extend(auto_qna)
        
        return {
            "categories": categories,
            "qa_pairs": qa_pairs,
            "last_updated": datetime.now().isoformat(),
            "raw_text": text[:1000] + "..." if len(text) > 1000 else text
        }
    
    def extract_firm_overview(self, text: str) -> List[str]:
        """Extract firm overview information"""
        patterns = [
            r"(?i).*(?:law firm|legal|attorney|practice|company|organization).*",
            r"(?i).*(?:professionals?|lawyers?|partners?|employees?).*\d+.*",
            r"(?i).*(?:full.?service|top.?tier|leading|established|founded).*",
            r"(?i).*(?:clients?|solutions?|services?|expertise).*",
            r"(?i).*(?:years?|decade|century).*(?:experience|history).*"
        ]
        
        sentences = re.split(r'[.!?]+', text)
        overview = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Avoid very short sentences
                for pattern in patterns:
                    if re.search(pattern, sentence):
                        overview.append(sentence)
                        break
        
        return list(set(overview))[:10]  # Remove duplicates, limit to 10
    
    def extract_offices(self, text: str) -> List[str]:
        """Extract office locations"""
        office_patterns = [
            r"(?i)(?:office|branch|location|presence).*?(?:in|at|across)\s+([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)",
            r"(?i)(Mumbai|Delhi|Bangalore|Chennai|Kolkata|Ahmedabad|Pune|Singapore|Hyderabad|Gurgaon|New York|London|Dubai|Hong Kong)",
            r"(?i)([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)\s+office",
        ]
        
        offices = []
        for pattern in office_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    offices.extend([city.strip() for city in match[0].split(',')])
                else:
                    offices.append(match.strip())
        
        return list(set(offices))
    
    def extract_programs(self, text: str) -> List[Dict[str, str]]:
        """Extract programs and initiatives"""
        programs = []
        
        # Enhanced program patterns
        program_patterns = [
            r"(?i)([A-Z][A-Z\s&\-]+(?:Program|Initiative|Project|Campaign))\s*[:\-]?\s*([^.!?]{10,}[.!?])",
            r"(?i)((?:training|workshop|seminar|conference|event)[^.!?]*[.!?])",
            r"(?i)((?:program|initiative|project|campaign)[^.!?]*[.!?])",
            r"(?i)([A-Z][A-Za-z\s]+(?:Awards?|Recognition|Competition))\s*[:\-]?\s*([^.!?]{10,}[.!?])"
        ]
        
        for pattern in program_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    programs.append({
                        "name": match[0].strip(),
                        "description": match[1].strip()
                    })
                else:
                    programs.append({
                        "name": "Initiative",
                        "description": match.strip() if isinstance(match, str) else str(match)
                    })
        
        return programs[:15]  # Limit to 15 programs
    
    def extract_technology(self, text: str) -> List[Dict[str, str]]:
        """Extract technology and tools information"""
        tech_tools = []
        
        tech_patterns = [
            r"(?i)([A-Z][a-zA-Z\s]+(?:AI|Technology|Platform|Tool|System|Software|App|Digital))\s*[:\-]?\s*([^.!?]*[.!?])",
            r"(?i)((?:artificial intelligence|machine learning|automation|digital|cloud|mobile|web)[^.!?]*[.!?])",
            r"(?i)((?:technology|platform|tool|system|software|application)[^.!?]*[.!?])"
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    tech_tools.append({
                        "name": match[0].strip(),
                        "description": match[1].strip()
                    })
                else:
                    tech_tools.append({
                        "name": "Technology",
                        "description": match.strip() if isinstance(match, str) else str(match)
                    })
        
        return tech_tools[:10]
    
    def extract_awards(self, text: str) -> List[str]:
        """Extract awards and recognition"""
        award_patterns = [
            r"(?i).*(?:award|recognition|winner|finalist|ranked|honored|received|achieved).*",
            r"(?i).*(?:named|recognized|awarded|celebrated|distinguished).*",
            r"(?i).*(?:top|best|leading|excellence|outstanding|prestigious).*"
        ]
        
        sentences = re.split(r'[.!?]+', text)
        awards = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:
                for pattern in award_patterns:
                    if re.search(pattern, sentence):
                        awards.append(sentence)
                        break
        
        return list(set(awards))[:8]
    
    def extract_practice_areas(self, text: str) -> List[str]:
        """Extract practice areas"""
        practice_areas = []
        
        # Enhanced practice areas
        common_areas = [
            "Corporate Law", "Banking", "Finance", "Litigation", "Intellectual Property",
            "Tax", "Employment", "Real Estate", "Regulatory", "Compliance",
            "Mergers", "Acquisitions", "Securities", "Capital Markets",
            "Infrastructure", "Energy", "Healthcare", "Technology", "Insurance",
            "Dispute Resolution", "Commercial", "Contract Law", "Environmental",
            "Immigration", "Family Law", "Criminal Law", "International Law"
        ]
        
        text_lower = text.lower()
        for area in common_areas:
            if area.lower() in text_lower:
                practice_areas.append(area)
        
        return practice_areas
    
    def extract_general_info(self, text: str) -> List[str]:
        """Extract other general information"""
        info_patterns = [
            r"(?i).*(?:established|founded|since|history|began|started).*",
            r"(?i).*(?:mission|vision|values|purpose|objective|goal).*",
            r"(?i).*(?:network|global|international|worldwide|across).*",
            r"(?i).*(?:commitment|dedicated|focus|specializ|expertise).*",
            r"(?i).*(?:culture|environment|team|community|collaboration).*"
        ]
        
        sentences = re.split(r'[.!?]+', text)
        general_info = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                for pattern in info_patterns:
                    if re.search(pattern, sentence):
                        general_info.append(sentence)
                        break
        
        return list(set(general_info))[:10]
    
    def extract_people_contacts(self, text: str) -> List[Dict[str, str]]:
        """Extract people and contact information"""
        people = []
        
        # Pattern for names with titles
        name_patterns = [
            r"(?i)([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[,:]?\s*(Partner|Director|Manager|Associate|Senior|Junior|Head|CEO|CTO|CFO|President)",
            r"(?i)(Mr\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][a-z]+ [A-Z][a-z]+)",
            r"(?i)([A-Z][a-z]+ [A-Z][a-z]+)\s*[,:]?\s*([\w\s]+@[\w\s]+)"
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    people.append({
                        "name": match[0].strip(),
                        "title": match[1].strip()
                    })
        
        return people[:10]
    
    def extract_events_news(self, text: str) -> List[Dict[str, str]]:
        """Extract events and news"""
        events = []
        
        event_patterns = [
            r"(?i)((?:conference|seminar|workshop|event|meeting|webinar|training)[^.!?]*[.!?])",
            r"(?i)((?:announcement|news|update|launch|release)[^.!?]*[.!?])",
            r"(?i)([A-Z][a-zA-Z\s]+(?:Event|Conference|Seminar|Workshop))\s*[:\-]?\s*([^.!?]*[.!?])"
        ]
        
        for pattern in event_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    events.append({
                        "title": match[0].strip(),
                        "description": match[1].strip()
                    })
                else:
                    events.append({
                        "title": "Event",
                        "description": match.strip() if isinstance(match, str) else str(match)
                    })
        
        return events[:8]
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy instead of nltk to avoid punkt_tab error"""
        doc = self.spacy_nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
    
    def generate_qna_from_sentences_with_t5(self, text: str, max_qna: int = 100) -> List[Dict[str, str]]:
        """Generate Q&A from raw text using T5 model"""
        sentences = self.split_sentences(text)
        qa_pairs = []
        count = 0

        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) < 5:
                continue

            # Prepare input for T5
            input_text = f"generate question: {sent} </s>"
            input_ids = self.qg_tokenizer.encode(input_text, return_tensors="pt")

            # Generate output
            output = self.qg_model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
            question = self.qg_tokenizer.decode(output[0], skip_special_tokens=True)

            if question:
                qa_pairs.append({
                    "question": question.strip(),
                    "answer": sent
                })
                count += 1

            if count >= max_qna:
                break

        return qa_pairs
    
    def generate_comprehensive_qa_pairs(self, categories: Dict[str, Any], full_text: str) -> List[Dict[str, str]]:
        """Generate comprehensive Q&A pairs from categorized data"""
        qa_pairs = []
        
        # Basic information questions
        qa_pairs.extend([
            {"question": "What is this document about?", "answer": self.get_document_summary(full_text)},
            {"question": "Tell me about the main content", "answer": self.get_document_summary(full_text)},
            {"question": "What information is available?", "answer": self.get_document_summary(full_text)},
            {"question": "Give me an overview", "answer": self.get_document_summary(full_text)},
            {"question": "What can you tell me about this?", "answer": self.get_document_summary(full_text)},
        ])
        
        # Firm overview questions
        if categories["firm_overview"]:
            overview_text = " ".join(categories["firm_overview"][:3])
            qa_pairs.extend([
                {"question": "What is the firm about?", "answer": overview_text},
                {"question": "Tell me about the company", "answer": overview_text},
                {"question": "What does the firm do?", "answer": overview_text},
                {"question": "Company information", "answer": overview_text},
                {"question": "About the organization", "answer": overview_text},
            ])
        
        # Location questions
        if categories["offices_locations"]:
            locations_text = f"The offices are located in: {', '.join(categories['offices_locations'])}"
            qa_pairs.extend([
                {"question": "Where are the offices located?", "answer": locations_text},
                {"question": "Office locations", "answer": locations_text},
                {"question": "Where can I find offices?", "answer": locations_text},
                {"question": "Branch locations", "answer": locations_text},
                {"question": "Geographic presence", "answer": locations_text},
            ])
        
        # Programs and initiatives
        if categories["programs_initiatives"]:
            programs_text = ". ".join([f"{p['name']}: {p['description']}" for p in categories["programs_initiatives"][:3]])
            qa_pairs.extend([
                {"question": "What programs are available?", "answer": programs_text},
                {"question": "Tell me about programs", "answer": programs_text},
                {"question": "What initiatives exist?", "answer": programs_text},
                {"question": "Available programs", "answer": programs_text},
                {"question": "Training programs", "answer": programs_text},
            ])
        
        # Technology questions
        if categories["technology_tools"]:
            tech_text = ". ".join([f"{t['name']}: {t['description']}" for t in categories["technology_tools"][:3]])
            qa_pairs.extend([
                {"question": "What technology is used?", "answer": tech_text},
                {"question": "Technology tools", "answer": tech_text},
                {"question": "Digital platforms", "answer": tech_text},
                {"question": "Technical solutions", "answer": tech_text},
                {"question": "Software and tools", "answer": tech_text},
            ])
        
        # Awards and recognition
        if categories["awards_recognition"]:
            awards_text = " ".join(categories["awards_recognition"][:3])
            qa_pairs.extend([
                {"question": "What awards have been received?", "answer": awards_text},
                {"question": "Recognition and awards", "answer": awards_text},
                {"question": "Achievements", "answer": awards_text},
                {"question": "Honors received", "answer": awards_text},
                {"question": "Awards and recognition", "answer": awards_text},
            ])
        
        # Practice areas
        if categories["practice_areas"]:
            areas_text = f"The practice areas include: {', '.join(categories['practice_areas'])}"
            qa_pairs.extend([
                {"question": "What are the practice areas?", "answer": areas_text},
                {"question": "Areas of expertise", "answer": areas_text},
                {"question": "Legal specializations", "answer": areas_text},
                {"question": "Service areas", "answer": areas_text},
                {"question": "What services are offered?", "answer": areas_text},
            ])
        
        # People and contacts
        if categories["people_contacts"]:
            people_text = ". ".join([f"{p['name']}: {p['title']}" for p in categories["people_contacts"][:3]])
            qa_pairs.extend([
                {"question": "Who are the key people?", "answer": people_text},
                {"question": "Staff information", "answer": people_text},
                {"question": "Team members", "answer": people_text},
                {"question": "Contact persons", "answer": people_text},
                {"question": "Leadership team", "answer": people_text},
            ])
        
        # Events and news
        if categories["events_news"]:
            events_text = ". ".join([f"{e['title']}: {e['description']}" for e in categories["events_news"][:3]])
            qa_pairs.extend([
                {"question": "What events are happening?", "answer": events_text},
                {"question": "Recent news", "answer": events_text},
                {"question": "Upcoming events", "answer": events_text},
                {"question": "Latest updates", "answer": events_text},
                {"question": "News and announcements", "answer": events_text},
            ])
        
        # General information
        if categories["general_info"]:
            general_text = " ".join(categories["general_info"][:3])
            qa_pairs.extend([
                {"question": "General information", "answer": general_text},
                {"question": "Additional details", "answer": general_text},
                {"question": "Other information", "answer": general_text},
                {"question": "More details", "answer": general_text},
                {"question": "Background information", "answer": general_text},
            ])
        
        return qa_pairs
    
    def get_document_summary(self, text: str) -> str:
        """Generate a summary of the document"""
        sentences = re.split(r'[.!?]+', text)
        # Get first few meaningful sentences
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 30][:3]
        return " ".join(meaningful_sentences)
    
    def search_raw_text(self, question: str) -> Optional[str]:
        """Search for relevant content in raw text as fallback"""
        if not self.knowledge_base.get("raw_text"):
            return None
            
        # Simple keyword matching
        question_lower = question.lower()
        raw_text = self.knowledge_base["raw_text"].lower()
        
        # Extract keywords from question
        keywords = [word for word in question_lower.split() if len(word) > 3]
        
        # Find sentences containing keywords
        sentences = re.split(r'[.!?]+', self.knowledge_base["raw_text"])
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            clean_sentences = [
                re.sub(r'\s+', ' ', sentence).strip().replace("..", ".").replace(" .", ".")
                for sentence in relevant_sentences[:2]
            ]
            return "🧾 Here's what I found:\n\n" + " ".join(clean_sentences)
        
        return None
    
    def load_knowledge_base(self) -> Dict[str, Any]:
        """Load existing knowledge base"""
        if os.path.exists(self.json_file_path):
            try:
                with open(self.json_file_path, 'r', encoding='utf-8') as f:
                    kb = json.load(f)
                    # Convert embeddings back to numpy array if they exist
                    if "embeddings" in kb and kb["embeddings"]:
                        kb["embeddings"] = np.array(kb["embeddings"])
                    return kb
            except Exception as e:
                print(f"Error loading knowledge base: {e}")
        
        # Return proper structure with empty lists instead of empty dict
        return {
            "categories": {},
            "qa_pairs": [],
            "embeddings": [],
            "last_updated": None,
            "raw_text": ""
        }
    
    def save_knowledge_base(self, knowledge: Dict[str, Any]):
        """Save knowledge base to JSON file"""
        try:
            # Convert embeddings to list for JSON serialization
            knowledge_to_save = knowledge.copy()
            if "embeddings" in knowledge_to_save and hasattr(knowledge_to_save["embeddings"], 'tolist'):
                knowledge_to_save["embeddings"] = knowledge_to_save["embeddings"].tolist()
            elif "embeddings" in knowledge_to_save and isinstance(knowledge_to_save["embeddings"], np.ndarray):
                knowledge_to_save["embeddings"] = knowledge_to_save["embeddings"].tolist()
            
            with open(self.json_file_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_to_save, f, indent=2, ensure_ascii=False)
            
            print(f"Knowledge base saved with {len(knowledge.get('qa_pairs', []))} Q&A pairs")
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
    
    def update_knowledge_base(self, new_knowledge: Dict[str, Any]):
        """Update existing knowledge base with new data"""
        # Replace all data with new data (since it's from the same document)
        self.knowledge_base = new_knowledge
        
        # Update embeddings
        self.update_embeddings()
        
        # Save updated knowledge base
        self.save_knowledge_base(self.knowledge_base)
    
    def update_embeddings(self):
        """Update embeddings for all Q&A pairs"""
        if self.knowledge_base["qa_pairs"]:
            questions = [qa["question"] for qa in self.knowledge_base["qa_pairs"]]
            try:
                embeddings = self.model.encode(questions)
                self.knowledge_base["embeddings"] = embeddings  # Keep as numpy array
                print(f"Generated embeddings for {len(questions)} questions")
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                self.knowledge_base["embeddings"] = []
    
    def find_similar_question(self, user_question: str, threshold: float = 0.3) -> Optional[Dict[str, Any]]:
        """Find the most semantically similar question using top-N search"""
        
        # Handle greetings/casual
        greeting_response = self.is_greeting_or_casual(user_question)
        if greeting_response:
            return {
                "question": user_question,
                "answer": greeting_response,
                "similarity": 1.0,
                "type": "greeting"
            }

        if not self.knowledge_base.get("qa_pairs"):
            return None

        embeddings = self.knowledge_base.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            print("No embeddings found, generating new ones...")
            self.update_embeddings()
            embeddings = self.knowledge_base.get("embeddings")

        if embeddings is None or len(embeddings) == 0:
            print("Failed to generate embeddings")
            return None

        try:
            # Encode and normalize user question
            cleaned_question = user_question.strip().lower()
            user_embedding = self.model.encode([cleaned_question])
            embeddings = np.array(embeddings)

            # Compute cosine similarity
            similarities = cosine_similarity(user_embedding, embeddings)[0]

            # Get top 3 matches
            top_n = 3
            top_indices = similarities.argsort()[-top_n:][::-1]

            print(f"User question: {user_question}")
            print("Top matches:")
            for idx in top_indices:
                print(f"Q: {self.knowledge_base['qa_pairs'][idx]['question']} -> similarity: {similarities[idx]}")

            # Return first highly similar answer (>= 0.7)
            for idx in top_indices:
                if similarities[idx] >= 0.7:
                    result = self.knowledge_base["qa_pairs"][idx].copy()
                    result["similarity"] = float(similarities[idx])
                    result["type"] = "document"
                    return result

            # If no strong match, return the best one (even if < 0.7)
            best_idx = top_indices[0]
            if similarities[best_idx] >= threshold:
                result = self.knowledge_base["qa_pairs"][best_idx].copy()
                result["similarity"] = float(similarities[best_idx])
                result["type"] = "document"
                return result

        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            return None

        return None
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Complete pipeline: extract, save, categorize, and update knowledge base"""
        try:
            print(f"Starting PDF processing for: {pdf_path}")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            print(f"Extracted text length: {len(text)} characters")
            
            if not text or len(text.strip()) < 100:
                return {
                    "success": False,
                    "message": "PDF appears to be empty or text extraction failed"
                }
            
            # Save text to file
            self.save_text_to_file(text, pdf_path)
            
            # Categorize text
            new_knowledge = self.categorize_text(text)
            print(f"Generated {len(new_knowledge['qa_pairs'])} Q&A pairs")
            
            # Update knowledge base
            self.update_knowledge_base(new_knowledge)
            
            return {
                "success": True,
                "message": f"Successfully processed PDF: {os.path.basename(pdf_path)}",
                "categories_found": len(new_knowledge["categories"]),
                "qa_pairs_generated": len(new_knowledge["qa_pairs"]),
                "document_name": os.path.basename(pdf_path),
                "text_length": len(text)
            }
            
        except Exception as e:
            print(f"Error in process_pdf: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing PDF: {str(e)}"
            }

# FastAPI Application
app = FastAPI(title="PDF Knowledge Base Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the knowledge extractor
extractor = PDFKnowledgeExtractor()

class ProcessPDFRequest(BaseModel):
    pdf_path: str

class QueryRequest(BaseModel):
    question: str
    threshold: Optional[float] = 0.3

@app.post("/process-pdf")
async def process_pdf(request: ProcessPDFRequest):
    """Process a PDF and update the knowledge base"""
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    result = extractor.process_pdf(request.pdf_path)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result

@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base with enhanced NLP"""
    print(f"Received query: {request.question}")
    
    # Debug: Check if knowledge base has data
    print(f"Knowledge base has {len(extractor.knowledge_base.get('qa_pairs', []))} Q&A pairs")
    print(f"Knowledge base has {len(extractor.knowledge_base.get('embeddings', []))} embeddings")
    
    result = extractor.find_similar_question(request.question, request.threshold)
    
    if result:
        return {
            "success": True,
            "question": result["question"],
            "answer": result["answer"],
            "similarity": result.get("similarity", 0),
            "type": result.get("type", "document")
        }
    else:
        # If no result found, try to find any relevant content from raw text
        fallback_answer = extractor.search_raw_text(request.question)
        if fallback_answer:
            return {
                "success": True,
                "question": request.question,
                "answer": fallback_answer,
                "similarity": 0.5,
                "type": "fallback"
            }
        else:
            return {
                "success": False,
                "message": "I couldn't find a relevant answer in the document. Could you please rephrase your question or ask about something specific from the document?",
                "suggestion": "Try asking about topics like company information, locations, programs, or services mentioned in the document.",
                "available_topics": list(extractor.knowledge_base.get("categories", {}).keys())
            }

@app.get("/knowledge-base")
async def get_knowledge_base():
    """Get the current knowledge base"""
    # Return knowledge base without embeddings for API response
    kb_copy = extractor.knowledge_base.copy()
    kb_copy["embeddings"] = []  # Remove embeddings from response
    return kb_copy

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "PDF Knowledge Base Chatbot API is running"}

@app.get("/debug")
async def debug_knowledge_base():
    """Debug endpoint to check knowledge base status"""
    return {
        "qa_pairs_count": len(extractor.knowledge_base.get("qa_pairs", [])),
        "embeddings_count": len(extractor.knowledge_base.get("embeddings", [])),
        "categories": list(extractor.knowledge_base.get("categories", {}).keys()),
        "sample_questions": [qa["question"] for qa in extractor.knowledge_base.get("qa_pairs", [])[:5]],
        "has_raw_text": bool(extractor.knowledge_base.get("raw_text")),
        "raw_text_length": len(extractor.knowledge_base.get("raw_text", "")),
        "last_updated": extractor.knowledge_base.get("last_updated")
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Knowledge Base Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "process_pdf": "/process-pdf",
            "query": "/query",
            "knowledge_base": "/knowledge-base",
            "debug": "/debug",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    # Auto-process the PDF on startup
    pdf_path = r"C:\Users\DELL\Desktop\Nishant Sir\khaitan_chatbot\Alumni Newsletter-KConnect APRIL 2025.pdf"
    
    print("Starting PDF Knowledge Base Chatbot...")
    
    # Initialize extractor first
    extractor = PDFKnowledgeExtractor()
    
    if os.path.exists(pdf_path):
        print(f"Processing PDF: {pdf_path}")
        result = extractor.process_pdf(pdf_path)
        print(f"Processing result: {result}")
        
        # Debug the knowledge base after processing
        print(f"Final knowledge base stats:")
        print(f"- Q&A pairs: {len(extractor.knowledge_base.get('qa_pairs', []))}")
        print(f"- Embeddings: {len(extractor.knowledge_base.get('embeddings', []))}")
        print(f"- Categories: {list(extractor.knowledge_base.get('categories', {}).keys())}")
        
        if extractor.knowledge_base.get('qa_pairs'):
            print("Sample questions:")
            for i, qa in enumerate(extractor.knowledge_base['qa_pairs'][:3]):
                print(f"  {i+1}. {qa['question']}")
    else:
        print(f"PDF file not found: {pdf_path}")
    
    # Move the extractor initialization before the FastAPI app
    # and remove the duplicate initialization after app creation
    
    uvicorn.run(app, host="0.0.0.0", port=8000)