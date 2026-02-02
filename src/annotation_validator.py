#!/usr/bin/env python3
"""
Annotation Validator for Egocentric Annotation Program
Validates annotations against Egocentric Annotation Program rules
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    ERROR = "ERROR"         # Audit fail condition - must fix
    WARNING = "WARNING"     # Should fix but not audit fail
    INFO = "INFO"           # Informational


@dataclass
class ValidationResult:
    """Result of annotation validation"""
    is_valid: bool
    issues: List[Dict]
    score: float = 100.0


class AnnotationValidator:
    """
    Validates annotations according to Egocentric Annotation Program rules
    Based on: https://audit.atlascapture.io/
    """

    def __init__(self, config: Dict):
        """
        Initialize validator with config rules

        Args:
            config: Configuration dict from config.yaml
        """
        self.egocentric_rules = config.get('egocentric_annotation', {})
        self.validation_config = config.get('validation', {})
        self.verb_rules = self.egocentric_rules.get('verbs', {})
        self.label_format = self.egocentric_rules.get('label_format', {})
        self.no_action_rules = self.egocentric_rules.get('no_action', {})
        self.granularity = self.egocentric_rules.get('granularity', {})

        # Forbidden verbs
        self.forbidden_verbs = set(self.verb_rules.get('forbidden', []))

        # Allowed verbs with definitions
        self.allowed_verbs = self.verb_rules.get('allowed', {})

        # Validation checks to run
        self.enable_checks = set(self.validation_config.get('checks', []))

    def validate_label(self, label: str, segment_id: str = "") -> ValidationResult:
        """
        Validate a single annotation label

        Args:
            label: The annotation text to validate
            segment_id: Optional segment identifier

        Returns:
            ValidationResult with issues and validity
        """
        issues = []

        # Skip if No Action
        if label.strip().lower() == "no action":
            return ValidationResult(is_valid=True, issues=[])

        # 1. Check forbidden verbs
        if 'forbidden_verbs' in self.enable_checks:
            issues.extend(self._check_forbidden_verbs(label, segment_id))

        # 2. Check for numerals
        if 'numerals' in self.enable_checks:
            issues.extend(self._check_numerals(label, segment_id))

        # 3. Check imperative voice
        if 'imperative_voice' in self.enable_checks:
            issues.extend(self._check_imperative_voice(label, segment_id))

        # 4. Check for object naming
        if 'object_naming' in self.enable_checks:
            issues.extend(self._check_object_naming(label, segment_id))

        # 5. Check verb compliance
        if 'verb_compliance' in self.enable_checks:
            issues.extend(self._check_verb_compliance(label, segment_id))

        # 6. Check length constraints
        issues.extend(self._check_length_constraints(label, segment_id))

        # 7. Check for intent-only language
        issues.extend(self._check_intent_only_language(label, segment_id))

        # Calculate score
        score = self._calculate_score(issues)

        # Determine validity
        error_count = sum(1 for i in issues if i['severity'] == ValidationSeverity.ERROR.value)
        is_valid = error_count == 0

        return ValidationResult(is_valid=is_valid, issues=issues, score=score)

    def validate_segment(self, segment: Dict) -> ValidationResult:
        """
        Validate a full segment including label and timestamps

        Args:
            segment: Dict with 'start_time', 'end_time', 'label', 'segment_id'

        Returns:
            ValidationResult with issues and validity
        """
        issues = []

        # Validate the label
        label_result = self.validate_label(segment.get('label', ''), segment.get('segment_id', ''))
        issues.extend(label_result.issues)

        # Validate timestamps if provided
        if 'timestamp_coverage' in self.enable_checks:
            issues.extend(self._check_timestamps(segment))

        # Check No Action rules
        if 'no_action_rules' in self.enable_checks:
            issues.extend(self._check_no_action_rules(segment))

        # Calculate overall score
        score = self._calculate_score(issues)

        # Determine validity
        error_count = sum(1 for i in issues if i['severity'] == ValidationSeverity.ERROR.value)
        is_valid = error_count == 0

        return ValidationResult(is_valid=is_valid, issues=issues, score=score)

    def validate_episode(self, annotations: List[Dict]) -> ValidationResult:
        """
        Validate a full episode of annotations

        Args:
            annotations: List of segment dicts

        Returns:
            ValidationResult with issues and overall validity
        """
        all_issues = []

        # Validate each segment
        for seg in annotations:
            result = self.validate_segment(seg)
            all_issues.extend(result.issues)

        # Check consistency across segments
        all_issues.extend(self._check_episode_consistency(annotations))

        # Check for dense/coarse mixing
        if 'dense_coarse_mixed' in self.enable_checks:
            all_issues.extend(self._check_dense_coarse_mixing(annotations))

        # Calculate overall score
        score = self._calculate_score(all_issues)

        # Determine validity
        error_count = sum(1 for i in all_issues if i['severity'] == ValidationSeverity.ERROR.value)
        is_valid = error_count == 0

        return ValidationResult(is_valid=is_valid, issues=all_issues, score=score)

    # ==================== Individual Check Methods ====================

    def _check_forbidden_verbs(self, label: str, segment_id: str) -> List[Dict]:
        """Check for forbidden verbs (inspect, check, examine, reach)"""
        issues = []

        label_lower = label.lower()

        for forbidden_verb in self.forbidden_verbs:
            # Check if forbidden verb appears as a word (not part of another word)
            pattern = r'\b' + re.escape(forbidden_verb) + r'\b'
            if re.search(pattern, label_lower):
                severity = ValidationSeverity.ERROR
                issue = {
                    'type': 'forbidden_verb',
                    'severity': severity.value,
                    'segment_id': segment_id,
                    'message': f"Forbidden verb '{forbidden_verb}' found",
                    'label': label.strip(),
                    'suggestion': self._suggest_replacement(forbidden_verb, label)
                }
                issues.append(issue)

        return issues

    def _check_numerals(self, label: str, segment_id: str) -> List[Dict]:
        """Check for numerals (digits 0-9)"""
        issues = []

        # Check for digits in the label
        if re.search(r'\d', label):
            severity = ValidationSeverity.ERROR
            issue = {
                'type': 'numerals',
                'severity': severity.value,
                'segment_id': segment_id,
                'message': 'Numerals found in label (use words instead)',
                'label': label.strip(),
                'suggestion': self._convert_numerals_to_words(label)
            }
            issues.append(issue)

        return issues

    def _check_imperative_voice(self, label: str, segment_id: str) -> List[Dict]:
        """Check if label uses imperative voice (starts with verb, not -ing)"""
        issues = []

        label = label.strip()

        # Check if it starts with -ing verb (present participle)
        if re.match(r'^[a-z]*ing\b', label, re.IGNORECASE):
            severity = ValidationSeverity.ERROR
            issue = {
                'type': 'imperative_voice',
                'severity': severity.value,
                'segment_id': segment_id,
                'message': 'Label starts with present participle (-ing) - use imperative voice instead',
                'label': label,
                'suggestion': self._convert_to_imperative(label)
            }
            issues.append(issue)

        return issues

    def _check_object_naming(self, label: str, segment_id: str) -> List[Dict]:
        """Check if label contains an object reference"""
        issues = []

        # Simple check: look for common object patterns
        # Most actions should have an object
        label_lower = label.lower()

        # Skip if it's a movement action (but those typically have objects too)
        if any(word in label_lower for word in ['walk', 'move through', 'navigate']):
            return issues

        # Check if there's a noun following the verb
        # Very simplified check - real implementation would be more sophisticated
        words = label_lower.split()
        if len(words) < 2:
            severity = ValidationSeverity.WARNING
            issue = {
                'type': 'object_naming',
                'severity': severity.value,
                'segment_id': segment_id,
                'message': 'Label may be missing object reference',
                'label': label.strip(),
                'suggestion': 'Add object being interacted with (e.g., "pick up spoon" instead of "pick up")'
            }
            issues.append(issue)

        return issues

    def _check_verb_compliance(self, label: str, segment_id: str) -> List[Dict]:
        """Check if verbs comply with allowed verbs set"""
        issues = []

        # Get first word (should be the verb)
        words = label.strip().split()
        if not words:
            return issues

        first_word = words[0].lower()

        # Check if it's an allowed verb
        allowed_verbs_list = list(self.allowed_verbs.keys())
        if first_word not in allowed_verbs_list:
            # Could be a generic verb not in our list - warning not error
            severity = ValidationSeverity.WARNING
            issue = {
                'type': 'verb_compliance',
                'severity': severity.value,
                'segment_id': segment_id,
                'message': f"First word '{first_word}' not in allowed verbs list",
                'label': label.strip(),
                'allowed_verbs': allowed_verbs_list
            }
            issues.append(issue)

        # Special check: if using "place", ensure there's a location
        if first_word == 'place':
            location_keywords = ['on', 'in', 'at', 'to', 'from']
            has_location = any(kw in label_lower for kw in location_keywords)
            if not has_location:
                severity = ValidationSeverity.ERROR
                issue = {
                    'type': 'verb_compliance',
                    'severity': severity.value,
                    'segment_id': segment_id,
                    'message': "Label uses 'place' but missing location (e.g., 'on table', 'in bin')",
                    'label': label.strip()
                }
                issues.append(issue)

        return issues

    def _check_no_action_rules(self, segment: Dict) -> List[Dict]:
        """Check No Action usage rules"""
        issues = []

        label = segment.get('label', '').strip()
        segment_id = segment.get('segment_id', '')

        if label.lower() == 'no action':
            # Rule: Do not combine "No Action" with real actions
            if len(label.split()) > 2:  # More than just "No Action"
                severity = ValidationSeverity.ERROR
                issue = {
                    'type': 'no_action_rules',
                    'severity': severity.value,
                    'segment_id': segment_id,
                    'message': 'No Action label combined with other text',
                    'label': label
                }
                issues.append(issue)

        return issues

    def _check_length_constraints(self, label: str, segment_id: str) -> List[Dict]:
        """Check if label meets length constraints"""
        issues = []

        text_constraints = self.egocentric_rules.get('text_constraints', {})
        max_words = text_constraints.get('max_words', 50)
        max_chars = text_constraints.get('max_chars', 300)

        word_count = len(label.split())
        char_count = len(label)

        if word_count > max_words:
            severity = ValidationSeverity.WARNING
            issue = {
                'type': 'length_constraints',
                'severity': severity.value,
                'segment_id': segment_id,
                'message': f'Label exceeds word limit ({word_count}/{max_words})',
                'label': label.strip()
            }
            issues.append(issue)

        if char_count > max_chars:
            severity = ValidationSeverity.WARNING
            issue = {
                'type': 'length_constraints',
                'severity': severity.value,
                'segment_id': segment_id,
                'message': f'Label exceeds character limit ({char_count}/{max_chars})',
                'label': label.strip()
            }
            issues.append(issue)

        return issues

    def _check_intent_only_language(self, label: str, segment_id: str) -> List[Dict]:
        """Check for intent-only language instead of physical actions"""
        issues = []

        intent_phrases = [
            'preparing to',
            'getting ready to',
            'about to',
            'planning to',
            'thinking about'
        ]

        label_lower = label.lower()

        for phrase in intent_phrases:
            if phrase in label_lower:
                severity = ValidationSeverity.ERROR
                issue = {
                    'type': 'intent_only_language',
                    'severity': severity.value,
                    'segment_id': segment_id,
                    'message': f"Intent-only language found: '{phrase}'. Prefer physical verbs.",
                    'label': label.strip()
                }
                issues.append(issue)

        return issues

    def _check_timestamps(self, segment: Dict) -> List[Dict]:
        """Check timestamp validity"""
        issues = []

        start = segment.get('start_time')
        end = segment.get('end_time')

        if start is None or end is None:
            # Missing timestamps
            severity = ValidationSeverity.WARNING
            issue = {
                'type': 'timestamp_coverage',
                'severity': severity.value,
                'segment_id': segment.get('segment_id', ''),
                'message': 'Missing timestamps'
            }
            issues.append(issue)
        elif end <= start:
            # Invalid: end before or equal to start
            severity = ValidationSeverity.ERROR
            issue = {
                'type': 'timestamp_coverage',
                'severity': severity.value,
                'segment_id': segment.get('segment_id', ''),
                'message': f'Invalid timestamps: end ({end}) <= start ({start})'
            }
            issues.append(issue)

        return issues

    def _check_episode_consistency(self, annotations: List[Dict]) -> List[Dict]:
        """Check consistency across episode segments"""
        issues = []

        # Extract objects used across segments
        objects_used = {}
        for seg in annotations:
            label = seg.get('label', '').strip().lower()
            # Simple object extraction - very basic
            words = label.split()
            if len(words) > 1:
                potential_objects = words[1:]  # After verb
                for obj in potential_objects:
                    if obj not in objects_used:
                        objects_used[obj] = []
                    objects_used[obj].append(seg.get('segment_id', ''))

        # Check for inconsistent naming (same thing called different names)
        # This is a simplified check - real implementation would need NLP

        return issues

    def _check_dense_coarse_mixing(self, annotations: List[Dict]) -> List[Dict]:
        """Check if dense and coarse labels are mixed incorrectly"""
        issues = []

        # Identify granularities
        granularities = []
        for seg in annotations:
            label = seg.get('label', '')
            is_dense = self._is_dense_label(label)
            granularities.append(is_dense)

        # Check if segment is supposed to be unified but mixed
        # For now, this is a placeholder - implement logic based on specific rules

        return issues

    def _is_dense_label(self, label: str) -> bool:
        """Determine if label is dense or coarse"""
        # Dense labels typically have multiple verbs separated by commas or "and"
        # Coarse labels typically have single verb
        comma_count = label.count(',')
        and_count = label.lower().count(' and ')

        # If multiple actions, likely dense
        if comma_count > 0 or and_count > 0:
            return True

        return False

    # ==================== Helper Methods ====================

    def _suggest_replacement(self, forbidden_verb: str, label: str) -> str:
        """Suggest replacement for forbidden verb"""
        replacements = {
            'inspect': 'adjust',
            'check': 'adjust',
            'examine': 'adjust',
            'reach': 'pick up'  # When appropriate
        }

        if forbidden_verb in replacements:
            return label.replace(forbidden_verb, replacements[forbidden_verb])

        return label

    def _convert_numerals_to_words(self, label: str) -> str:
        """Convert numerals to words (simple implementation)"""
        # Very basic - full implementation would need comprehensive number-to-words
        number_words = {
            '0': 'zero',
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine'
        }

        result = label
        for digit, word in number_words.items():
            result = result.replace(digit, word)

        return result

    def _convert_to_imperative(self, label: str) -> str:
        """Convert present participle to imperative (simple)"""
        # Remove -ing and adjust
        # This is very simplified - real implementation needs full verb conjugation
        if label.endswith('ing'):
            # Remove -ing and return as is (not perfect but a start)
            return label[:-3]
        return label

    def _calculate_score(self, issues: List[Dict]) -> float:
        """Calculate validation score (0-100)"""
        total_deduction = 0

        for issue in issues:
            severity = issue.get('severity', 'INFO')
            if severity == ValidationSeverity.ERROR.value:
                total_deduction += 10
            elif severity == ValidationSeverity.WARNING.value:
                total_deduction += 5
            elif severity == ValidationSeverity.INFO.value:
                total_deduction += 1

        score = max(0, 100 - total_deduction)
        return round(score, 1)

    def get_validation_report(self, result: ValidationResult) -> str:
        """Generate a formatted validation report"""
        if result.is_valid:
            return "✅ Annotation PASSED validation\n"

        report = "❌ Annotation FAILED validation\n\n"

        # Group issues by severity
        errors = [i for i in result.issues if i['severity'] == ValidationSeverity.ERROR.value]
        warnings = [i for i in result.issues if i['severity'] == ValidationSeverity.WARNING.value]
        infos = [i for i in result.issues if i['severity'] == ValidationSeverity.INFO.value]

        if errors:
            report += f"🔴 ERRORS ({len(errors)}):\n"
            for i, error in enumerate(errors, 1):
                report += f"  {i}. {error['message']}\n"
                if 'label' in error:
                    report += f"     Label: \"{error['label']}\"\n"
                if 'suggestion' in error:
                    report += f"     Suggestion: {error['suggestion']}\n"
            report += "\n"

        if warnings:
            report += f"🟡 WARNINGS ({len(warnings)}):\n"
            for i, warning in enumerate(warnings, 1):
                report += f"  {i}. {warning['message']}\n"
                if 'label' in warning:
                    report += f"     Label: \"{warning['label']}\"\n"
            report += "\n"

        if infos:
            report += f"🔵 INFO ({len(infos)}):\n"
            for i, info in enumerate(infos, 1):
                report += f"  {i}. {info['message']}\n"

        report += f"\n📊 Score: {result.score}/100\n"

        return report


# ==================== Utility Functions ====================

def parse_atlas_annotation(line: str) -> Optional[Dict]:
    """
    Parse Atlas format annotation line: START_TIME-END_TIME#ID Description

    Args:
        line: Annotation line string

    Returns:
        Dict with 'start_time', 'end_time', 'segment_id', 'label' or None
    """
    try:
        # Match pattern: 0:00.0-0:20.0#1 Description
        match = re.match(r'(\d+:\d+\.\d+)-(\d+:\d+\.\d+)#(\d+)\s*(.*)', line)
        if match:
            start, end, seg_id, label = match.groups()
            return {
                'start_time': start,
                'end_time': end,
                'segment_id': seg_id,
                'label': label.strip()
            }
    except Exception as e:
        pass

    return None


def main():
    """Test the validator"""
    import yaml

    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create validator
    validator = AnnotationValidator(config)

    # Test labels
    test_labels = [
        "pick up three knives from table",
        "assembling black ballpoint pens",
        "inspect the tool carefully",
        "place cup",
        "Pick up cloth and wipe table",
        "No Action"
    ]

    print("Egocentric Annotation Validator\n")
    print("=" * 60)

    for i, label in enumerate(test_labels, 1):
        print(f"\n{i}. Testing: \"{label}\"")
        print("-" * 60)
        result = validator.validate_label(label, f"test_{i}")
        print(validator.get_validation_report(result))


if __name__ == '__main__':
    main()