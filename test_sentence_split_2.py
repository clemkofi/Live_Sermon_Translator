import re
import spacy
from typing import List, Tuple
import torch

class GermanTranslationSegmenter:
    def __init__(self, translation_method='marian'):
        # Load German spaCy model for linguistic analysis
        self.nlp = spacy.load('de_core_news_sm')
        
        # Translation method selection
        self.translation_method = translation_method
        
        if translation_method == 'marian':
            try:
                from transformers import MarianMTModel, MarianTokenizer
                model_name = 'Helsinki-NLP/opus-mt-de-en'
                self.tokenizer = MarianTokenizer.from_pretrained(model_name)
                self.translator = MarianMTModel.from_pretrained(model_name)
            except ImportError:
                print("Marian translator not available. Falling back to alternative method.")
                self.translation_method = 'googletrans'
        
        if translation_method == 'googletrans':
            try:
                from googletrans import Translator
                self.translator = Translator()
            except ImportError:
                print("googletrans not available. Please install it with 'pip install googletrans==3.1.0a0'")
                self.translation_method = None

    def _identify_segment_points(self, text: str) -> List[int]:
        """
        Identify optimal segmentation points in the text
        
        Criteria for segmentation:
        1. Full stops
        2. Specific conjunctions
        3. Comma-separated clauses
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Segmentation points
        segment_points = []
        
        # Predefined conjunctions to consider for potential breaks
        break_conjunctions = {
            'und', 'oder', 'aber', 'sondern',  # coordinating conjunctions
            'dass', 'weil', 'wenn', 'obwohl', 'falls'  # subordinating conjunctions
        }
        
        for sent in doc.sents:
            # Add full stop points
            segment_points.append(sent.end_char)
            
            # Check for additional break points within sentences
            for token in sent:
                # Break on specific conjunctions
                if token.text.lower() in break_conjunctions:
                    segment_points.append(token.idx + len(token.text))
        
        # Sort and remove duplicates
        return sorted(set(segment_points))

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into optimal segments
        """
        # Identify segment points
        segment_points = self._identify_segment_points(text)
        
        # Add start and end points
        segment_points = [0] + segment_points + [len(text)]
        
        # Create segments
        segments = []
        for i in range(len(segment_points) - 1):
            start = segment_points[i]
            end = segment_points[i+1]
            segment = text[start:end].strip()
            
            # Apply minimum and maximum segment length constraints
            if 10 < len(segment) < 250:
                segments.append(segment)
        
        return segments

    def translate_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Translate text with hybrid segmentation
        
        Returns list of tuples: (original segment, translated segment)
        """
        # Split text into segments
        segments = self._split_text(text)
        
        # Translate segments
        translations = []
        
        if self.translation_method == 'marian':
            for segment in segments:
                # Translate using Hugging Face Marian MT model
                inputs = self.tokenizer(segment, return_tensors="pt", padding=True)
                translated = self.tokenizer.decode(
                    self.translator.generate(**inputs)[0], 
                    skip_special_tokens=True
                )
                translations.append((segment, translated))
        
        elif self.translation_method == 'googletrans':
            for segment in segments:
                # Translate using googletrans
                translated = self.translator.translate(segment, src='de', dest='en').text
                translations.append((segment, translated))
        
        else:
            raise ValueError("No translation method available. Install either MarianMT or googletrans.")
        
        return translations

# Example usage
def main():
    # Try different translation methods
    try:
        # Try Marian translator first
        segmenter = GermanTranslationSegmenter(translation_method='marian')
    except Exception:
        # Fallback to googletrans
        segmenter = GermanTranslationSegmenter(translation_method='googletrans')
    
    # Example German text
    german_text = (
        "Heute ist ein wunderschöner Tag. "
        "Die Sonne scheint und die Vögel singen, "
        "während ich am Fenster sitze und nachdenke. "
        "Es ist wichtig, dass man die kleinen Momente im Leben genießt."
    )
    
    # Perform translation
    translations = segmenter.translate_text(german_text)
    
    # Print results
    print("Segmented Translations:")
    for original, translated in translations:
        print(f"DE: {original}")
        print(f"EN: {translated}")
        print("---")

if __name__ == "__main__":
    main()