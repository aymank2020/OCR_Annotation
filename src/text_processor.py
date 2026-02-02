"""
Text Processing Module for Easy Mode Compliance
Atlas Action Recognition
"""

import re
from typing import List, Dict, Set
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class EasyModeProcessor:
    """
    Process and validate action descriptions according to Easy Mode rules
    
    Easy Mode Rules:
    - 8-40 seconds duration
    - Descriptive sentences (not commands)
    - Present participle verbs (-ing)
    - "the" is allowed
    - Goal-oriented, not motion-based
    - No forbidden words (context, intent, judgment)
    """
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = self._default_config()
        
        self.min_duration = config.get('min_duration', 8)
        self.max_duration = config.get('max_duration', 40)
        self.max_words = config.get('max_words', 25)
        self.max_chars = config.get('max_chars', 200)
        self.use_ing_verbs = config.get('use_ing_verbs', True)
        self.allow_the = config.get('allow_the', True)
        self.forbidden_words = set(config.get('forbidden_words', []))
        self.goal_oriented = config.get('goal_oriented', True)
        
        self.lemmatizer = WordNetLemmatizer()
    
    def _default_config(self) -> Dict:
        """Default Easy Mode configuration"""
        return {
            'min_duration': 8,
            'max_duration': 40,
            'max_words': 25,
            'max_chars': 200,
            'use_ing_verbs': True,
            'allow_the': True,
            'forbidden_words': [
                'next', 'other', 'then', 'first', 'second',
                'carefully', 'trying', 'preparing to',
                'inspect', 'examine', 'check'
            ],
            'goal_oriented': True
        }
    
    def validate_text(self, text: str) -> Dict[str, any]:
        """
        Validate text against Easy Mode rules
        
        Returns:
            dict with validation results
        """
        errors = []
        warnings = []
        
        # Length checks
        word_count = len(text.split())
        char_count = len(text)
        
        if word_count > self.max_words:
            errors.append(f"Too many words: {word_count} (max: {self.max_words})")
        
        if char_count > self.max_chars:
            errors.append(f"Too long: {char_count} chars (max: {self.max_chars})")
        
        # Forbidden words check
        text_lower = text.lower()
        found_forbidden = []
        for word in self.forbidden_words:
            if word in text_lower:
                found_forbidden.append(word)
        
        if found_forbidden:
            errors.append(f"Forbidden words found: {', '.join(found_forbidden)}")
        
        # Present participle check
        if self.use_ing_verbs:
            words = word_tokenize(text.lower())
            # Check if first verb is -ing form
            has_ing_verb = any(word.endswith('ing') for word in words[:5])
            if not has_ing_verb:
                warnings.append("Should start with present participle verb (-ing)")
        
        # Descriptive check (not command)
        if text.strip() and text[0].isupper():
            # Good - starts with capital
            pass
        else:
            warnings.append("Should start with capital letter")
        
        # Check for command indicators
        command_words = ['pick up', 'place', 'put', 'take', 'move']
        if text.split()[0].lower() in command_words:
            warnings.append("Avoid command form - use present participle")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'word_count': word_count,
            'char_count': char_count
        }
    
    def process_text(self, text: str) -> str:
        """
        Process text to conform to Easy Mode
        
        Returns:
            Processed text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Remove forbidden words (replace with similar)
        replacements = {
            'next': '',
            'other': '',
            'then': 'and',
            'carefully': '',
            'trying to': '',
            'preparing to': '',
            'inspect': 'look at',
            'examine': 'look at',
            'check': 'look at'
        }
        
        text_lower = text.lower()
        for forbidden, replacement in replacements.items():
            if forbidden in text_lower:
                # Case-insensitive replace
                pattern = re.compile(re.escape(forbidden), re.IGNORECASE)
                text = pattern.sub(replacement, text)
        
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        return text
    
    def suggest_improvements(self, text: str) -> List[str]:
        """
        Suggest improvements for the text
        
        Returns:
            List of suggestions
        """
        suggestions = []
        
        validation = self.validate_text(text)
        
        # Add error-based suggestions
        for error in validation['errors']:
            if 'Too many words' in error:
                suggestions.append("Shorten description - focus on main action")
            elif 'Forbidden words' in error:
                suggestions.append("Remove context/intent words - describe what's visible")
        
        # Add warning-based suggestions
        for warning in validation['warnings']:
            if 'present participle' in warning.lower():
                suggestions.append("Start with -ing verb (e.g., 'Cleaning' not 'Clean')")
            elif 'capital letter' in warning.lower():
                suggestions.append("Start with capital letter")
        
        # Goal-oriented check
        motion_words = ['picking up', 'moving', 'placing', 'rotating', 'flipping']
        has_motion = any(word in text.lower() for word in motion_words)
        
        if has_motion and self.goal_oriented:
            suggestions.append("Describe the goal, not individual motions")
            suggestions.append("Example: 'Cleaning the sneaker' not 'Picking up sneaker, rotating, wiping'")
        
        return suggestions
    
    def extract_key_concepts(self, text: str) -> Dict[str, List[str]]:
        """
        Extract key concepts from text
        
        Returns:
            dict with verbs, objects, etc.
        """
        words = word_tokenize(text.lower())
        
        # Find -ing verbs (actions)
        actions = [w for w in words if w.endswith('ing') and len(w) > 4]
        
        # Find nouns (objects)
        # Simple heuristic: words after 'the', 'a', or between verbs
        objects = []
        for i, word in enumerate(words):
            if i > 0 and words[i-1] in ['the', 'a', 'an']:
                objects.append(word)
        
        return {
            'actions': actions,
            'objects': objects,
            'full_text': text
        }
    
    def generate_atlas_label(self, start: float, end: float, segment_id: int, text: str) -> str:
        """
        Generate Atlas format label
        
        Format: 0:00.0-0:20.0#1 Action description
        """
        start_min = int(start // 60)
        start_sec = start % 60
        end_min = int(end // 60)
        end_sec = end % 60
        
        # Process text
        text = self.process_text(text)
        
        return f"{start_min}:{start_sec:04.1f}-{end_min}:{end_sec:04.1f}#{segment_id} {text}"
    
    def parse_atlas_label(self, label: str) -> Dict:
        """
        Parse Atlas format label
        
        Returns:
            dict with start, end, id, text
        """
        match = re.match(r'(\d+):(\d+\.\d+)-(\d+):(\d+\.\d+)#(\d+)\s+(.+)', label)
        
        if not match:
            return None
        
        start_min, start_sec, end_min, end_sec, seg_id, text = match.groups()
        
        start_time = float(start_min) * 60 + float(start_sec)
        end_time = float(end_min) * 60 + float(end_sec)
        
        return {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time,
            'segment_id': int(seg_id),
            'text': text.strip()
        }


if __name__ == "__main__":
    # Test processor
    print("Testing EasyMode Text Processor...\n")
    
    processor = EasyModeProcessor()
    
    # Test cases
    test_texts = [
        "Dusting the upper body, laces, and heel of the black sneaker with a feather duster",
        "Cleaning the sole of the sneaker and placing it back on the display shelf",
        "Carefully pick up the next sneaker and inspect it",  # Bad - has forbidden words
        "Pick up sneaker, rotate, wipe, place down",  # Bad - motion-based
    ]
    
    for text in test_texts:
        print(f"Original: {text}")
        
        # Validate
        validation = processor.validate_text(text)
        print(f"Valid: {validation['valid']}")
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
        
        # Process
        processed = processor.process_text(text)
        print(f"Processed: {processed}")
        
        # Suggestions
        suggestions = processor.suggest_improvements(text)
        if suggestions:
            print(f"Suggestions: {suggestions}")
        
        # Atlas format
        atlas = processor.generate_atlas_label(0.0, 20.0, 1, text)
        print(f"Atlas: {atlas}")
        
        print("-" * 70)
    
    print("\n✅ Text processor test complete!")
