#############
# RAG V1
# Review word by word using MINEDU manual for Aymara language correction
# ###################

import json
import re
import time
import urllib.parse
import unicodedata
import requests
from typing import Dict, List
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

class RAGAymara:
    """RAG system for Aymara language correction using MINEDU manual"""
    
    def __init__(self, manual_path: str = 'data/libros/Aymara/minedu.json'):
        self.manual_path = manual_path
        self.documents = []
        self.retriever = None
        self.learned_suffixes = {}
        self.orthography_rules = {}
        self.base_vocabulary = self._load_base_vocabulary()
        self.rae_cache = {}
        
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.generator = pipeline("text-generation",
            model="microsoft/DialoGPT-medium",
            device=0 if torch.cuda.is_available() else -1,
            pad_token_id=50256
        )
        
        self._learn_from_manual()
    
    def _load_base_vocabulary(self):
        with open("data/diccionarios/aymara/dic-2-ay-es-simple.json", "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        
        vocabulary = {}
        for entry in vocab_json:
            if "Aymara" in entry and isinstance(entry["Aymara"], str):
                word = entry["Aymara"].strip().lower()
                word = re.sub(r"[!¡?.;,:'\"()\[\]{}<>]", "", word)
                word = unicodedata.normalize("NFC", word)
                if len(word) > 2:
                    meaning = entry.get("Español", "").strip()
                    vocabulary[word] = {'meaning': meaning}
        return vocabulary
    
    def _learn_from_manual(self):
        with open(self.manual_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        content = data.get("contenido", [])
        current_context = []
        
        for item in content:
            item_type = item.get("tipo", "")
            text = item.get("contenido", "").strip()
            level = item.get("nivel", 1)
            
            if item_type == "encabezado":
                if len(current_context) >= level:
                    current_context = current_context[:level-1]
                current_context.append(text)
                continue
            
            if len(text) > 10:
                context = " > ".join(current_context)
                self.documents.append(Document(page_content=text, metadata={"context": context}))
                self._extract_rules(text, context)
        
        if self.documents:
            self.retriever = BM25Retriever.from_documents(self.documents)
    
    def _extract_rules(self, text, context):
        text_lower = text.lower()
        
        specific_suffixes = [
            ('tha', 'ablative'), ('phana', 'imperative'), ('spha', 'junction'), ('muku', 'general'),
            ('ñacha', 'desiderative'), ('sma', 'transitional'), ('wi', 'general'), ('xa', 'general'),
            ('wa', 'general'), ('mpi', 'general'), ('ña', 'general'), ('ta', 'general')
        ]
        
        for suffix, function in specific_suffixes:
            if 'sufijo' in text_lower and suffix in text_lower:
                self.learned_suffixes[suffix] = {
                    'context': context, 'function': function, 'rule': text[:200]
                }
    
    def _is_aymara(self, word):
        if len(word) < 3:
            return False
        clean_word = re.sub(r'[^a-záäüöñ\']', '', word.lower())
        indicators = [
            any(c in clean_word for c in 'ñäüö'),
            any(seq in clean_word for seq in ['ph', 'th', 'kh', 'qh', "'"]),
            clean_word.endswith(('ña', 'ta', 'wi', 'xa', 'wa', 'tha', 'mpi')),
            clean_word.startswith(('x', 'q', 'w'))
        ]
        return sum(indicators) >= 2
    
    def verify_rae(self, word: str) -> bool:
        clean_word = re.sub(r'[^a-záéíóúüñ]', '', word.lower().strip())
        if len(clean_word) < 3 or self._is_aymara(clean_word):
            return False
            
        if clean_word in self.rae_cache:
            return self.rae_cache[clean_word]
        
        url = f"https://dle.rae.es/{urllib.parse.quote(clean_word)}"
        headers = {'User-Agent': 'Mozilla/5.0', 'Accept-Language': 'es-ES,es;q=0.9'}
        resp = requests.get(url, headers=headers, timeout=10)
        time.sleep(0.3)
        
        response_text = resp.text.lower()
        exists = (resp.status_code == 200 and 'article' in response_text and 
                 'id="resultados"' in response_text and
                 any(x in response_text for x in ['definición', 'significa', 'adj.', 'sust.']) and
                 not any(x in response_text for x in ['no se encontró', 'error']))
        
        self.rae_cache[clean_word] = exists
        return exists
    
    def is_fake_word(self, word: str) -> bool:
        return any([
            bool(re.search(r'(.)\1{3,}', word.lower())),
            any(seq in word.lower() for seq in ['qwerty', 'asdf', 'zxcv', '123']),
            bool(re.search(r'\d', word)),
            len(re.findall(r'[^a-záéíóúüäöñ\']', word)) > len(word) * 0.3
        ])
    
    def clean_word(self, word: str) -> str:
        clean = re.sub(r'[^a-záéíóúüäöñ\']', '', word.lower())
        return re.sub(r'(.)\1{3,}', r'\1\1', clean)
    
    def search_aymara_correction(self, word: str) -> Dict:
        original_word = word.lower()
        clean_word = self.clean_word(original_word)
        
        if len(clean_word) < 3:
            return None
        
        candidates = list(self.base_vocabulary.keys())
        best_candidate = None
        best_distance = float('inf')
        
        for candidate in candidates:
            if abs(len(clean_word) - len(candidate)) > 3:
                continue
            
            error_detected = self._detect_basic_error(clean_word, candidate)
            if error_detected:
                distance = self._levenshtein_distance(clean_word, candidate)
                if distance <= 3 and distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate
        
        if best_candidate:
            return {
                'word': best_candidate,
                'distance': best_distance,
                'word_info': self.base_vocabulary.get(best_candidate, {})
            }
        return None
    
    def _detect_basic_error(self, original, candidate):
        # Special characters
        if re.search(r'[^a-záéíóúüäöñ\']', original):
            clean_original = re.sub(r'[^a-záéíóúüäöñ\']', '', original)
            if clean_original == candidate:
                return True
        
        # Excessive repetition
        no_repetitions = re.sub(r'(.)\1{2,}', r'\1', original)
        if no_repetitions == candidate:
            return True
        
        # Extra characters
        if len(original) > len(candidate) and (original.startswith(candidate) or original.endswith(candidate)):
            return True
        
        # Missing characters  
        if len(candidate) > len(original) and candidate.startswith(original):
            return True
        
        # Simple substitution
        if len(original) == len(candidate):
            differences = sum(1 for a, b in zip(original, candidate) if a != b)
            if differences <= 2:
                return True
        
        return False
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + (c1 != c2)))
            prev_row = curr_row
        return prev_row[-1]
    
    def analyze_word(self, word: str) -> Dict:
        original_word = word
        word_lower = word.lower()
        
        if self.is_fake_word(original_word):
            return {'word': original_word, 'classification': 'fake_word'}
        
        if self.verify_rae(word_lower):
            return {'word': original_word, 'classification': 'spanish_loan'}
        
        if word_lower in self.base_vocabulary:
            word_info = self.base_vocabulary[word_lower]
            return {
                'word': original_word, 
                'classification': 'correct_aymara',
                'meaning': word_info.get('meaning', '')
            }
        
        # Check morphology
        for suffix in sorted(self.learned_suffixes.keys(), key=len, reverse=True):
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix):
                root = word_lower[:-len(suffix)]
                if root in self.base_vocabulary:
                    return {
                        'word': original_word,
                        'classification': 'aymara_morphology',
                        'root': root,
                        'suffix': suffix,
                        'root_meaning': self.base_vocabulary[root].get('meaning', ''),
                        'suffix_function': self.learned_suffixes[suffix].get('function', 'suffix')
                    }
        
        # Search for correction
        clean_word = self.clean_word(word_lower)
        if len(clean_word) >= 3:
            correction_result = self.search_aymara_correction(word_lower)
            if correction_result:
                return {
                    'word': original_word,
                    'classification': 'typographic_error',
                    'correction': correction_result['word'],
                    'distance': correction_result['distance'],
                    'correction_meaning': correction_result['word_info'].get('meaning', '')
                }
        
        return {'word': original_word, 'classification': 'unrecognized'}
    
    def generate_pedagogical_explanation(self, original_sentence, corrected_sentence, analysis):
        if not self.generator:
            return f"I see you wrote '{original_sentence}'. The correction to '{corrected_sentence}' follows MINEDU manual rules."
        
        # Prepare analysis info
        corrections = []
        morphology = []
        for result in analysis:
            word = result['word']
            classification = result['classification']
            
            if classification == 'typographic_error':
                correction = result.get('correction', '')
                corrections.append(f"'{word}' corrected to '{correction}'")
            elif classification == 'aymara_morphology':
                root = result.get('root', '')
                suffix = result.get('suffix', '')
                morphology.append(f"'{word}' = root '{root}' + suffix '-{suffix}'")
        
        # Get manual context
        manual_context = ""
        if self.retriever and (corrections or morphology):
            search_term = "ortografía escritura aymara"
            docs = self.retriever.get_relevant_documents(search_term)
            if docs:
                manual_context = docs[0].page_content[:100]
        
        prompt = f"""You are an Aymara teacher using MINEDU manual. Student wrote: "{original_sentence}". Corrected version: "{corrected_sentence}". 
                Analysis: {', '.join(corrections + morphology)}
                Manual context: {manual_context}
                Explain pedagogically:"""
        
        response = self.generator(
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=50256
        )[0]['generated_text']
        
        explanation = response[len(prompt):].strip()
        
        if len(explanation) < 30:
            return f"I see you wrote '{original_sentence}'. The correction to '{corrected_sentence}' follows MINEDU manual."
        
        return explanation[:500]
    
    def process_for_agent(self, sentence: str) -> Dict:
        words = re.findall(r"\b[\w'%&$#@!*+=<>?/\\|~`^{}[\]()]+\b", sentence)
        words = [w for w in words if len(w.strip()) >= 2]
        
        if not words:
            return {
                'original_sentence': sentence,
                'corrected_sentence': sentence,
                'has_changes': False,
                'complete_analysis': []
            }

        analysis = [self.analyze_word(w) for w in words]
        corrected_sentence = sentence
        
        # Apply corrections and removals
        for result in analysis:
            original_word = result['word']
            classification = result['classification']
            
            if classification == 'typographic_error':
                correction = result.get('correction', '')
                if correction:
                    corrected_sentence = corrected_sentence.replace(original_word, correction, 1)
            elif classification in ['unrecognized', 'fake_word']:
                pattern = r'\b' + re.escape(original_word) + r'\b'
                corrected_sentence = re.sub(pattern, '', corrected_sentence)
        
        corrected_sentence = ' '.join(corrected_sentence.split())
        has_changes = sentence != corrected_sentence
        
        return {
            'original_sentence': sentence,
            'corrected_sentence': corrected_sentence,
            'has_changes': has_changes,
            'complete_analysis': analysis
        }
    
    def get_sentences_for_agent(self, sentence: str) -> Dict[str, str]:
        result = self.process_for_agent(sentence)
        return {
            'original_sentence': result['original_sentence'],
            'corrected_sentence': result['corrected_sentence']
        }
    
    def analyze_sentence(self, sentence: str) -> str:
        processed_data = self.process_for_agent(sentence)
        
        result = [
            f"Original sentence: '{processed_data['original_sentence']}'",
            f"Corrected sentence: '{processed_data['corrected_sentence']}'",
            "DETAILED ANALYSIS BY WORD:"
        ]
        
        explanation = self.generate_pedagogical_explanation(
            processed_data['original_sentence'], 
            processed_data['corrected_sentence'], 
            processed_data['complete_analysis']
        )
        result.append(explanation)
        
        return '\n'.join(result)
