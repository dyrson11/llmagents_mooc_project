import re
from collections import Counter
from difflib import SequenceMatcher
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

USE_HUGGINGFACE = True

FILE_IN1 = "americasnlp2025/ST1_MachineTranslation/data/aymara-spanish/train.aym"
FILE_IN2 = "americasnlp2025/ST1_MachineTranslation/data/aymara-spanish/train.es"
FILE_OUT1 = "americasnlp2025/ST1_MachineTranslation/data/aymara-spanish/clean_a.aym"
FILE_OUT2 = "americasnlp2025/ST1_MachineTranslation/data/aymara-spanish/clean_e.es"

TOLERANCE_LEVEL = 100  # clean level: 100, 75, 50, 25

BASE_NAME_SIMILARITY_THRESHOLD = 0.8 # Bolivia, Boliviata , Ayentina ,Argentina, Cristina, Christina
BASE_WORD_TOLERANCE = 10
BASE_CHARACTER_TOLERANCE = 30

def adjust_thresholds_by_tolerance(level):
    """
    Adjusts thresholds based on tolerance level.
    """
    factor = level / 100.0
    similarity_threshold = max(0.4, BASE_NAME_SIMILARITY_THRESHOLD * factor)
    word_tol = int(BASE_WORD_TOLERANCE / factor) if factor > 0 else 100
    char_tol = int(BASE_CHARACTER_TOLERANCE / factor) if factor > 0 else 300
    
    return similarity_threshold, word_tol, char_tol

NAME_SIMILARITY_THRESHOLD, WORD_TOLERANCE, CHARACTER_TOLERANCE = adjust_thresholds_by_tolerance(TOLERANCE_LEVEL)

NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_and_count_numbers(line):
    number_pattern = re.compile(r'\d+')
    numbers = number_pattern.findall(line)
    return Counter(int(num) for num in numbers)

def extract_named_entities(text):
    if not USE_HUGGINGFACE:
        return extract_capitalized_words(text)
    
    processed_text = re.sub(r'[.!?]\s+([A-Z][\wáéíóúüñÁÉÍÓÚÜÑ]*)',
                            lambda m: m.group(0).replace(m.group(1), 'XXXXXX'), text)
    processed_text = re.sub(r'^\s*([A-Z][\wáéíóúüñÁÉÍÓÚÜÑ]*)',
                            lambda m: m.group(0).replace(m.group(1), 'XXXXXX'), processed_text)
    processed_text = re.sub(r'\s{2,}([A-Z][\wáéíóúüñÁÉÍÓÚÜÑ]*)',
                            lambda m: m.group(0).replace(m.group(1), 'XXXXXX'), processed_text)
    
    try:
        entities = ner_pipeline(processed_text)
        relevant_types = {"PER", "ORG", "LOC", "PERSON", "ORGANIZATION", "LOCATION", 
                          "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"}
        filtered = []
        for ent in entities:
            if ent['entity_group'] in relevant_types and ent['word'] != 'XXXXXX':
                word = ent['word']
                if len(word) >= 2 and 'XXXXX' not in word:
                    filtered.append(word)
        return filtered
    except Exception:
        return extract_capitalized_words(text)

def extract_capitalized_words(text):
    common_words = {
        "El", "La", "Los", "Las", "Un", "Una", "Unos", "Unas", "Yo", "Tú", "Él", "Ella",
        "Nosotros", "Nosotras", "Vosotros", "Vosotras", "Ellos", "Ellas", "Este", "Esta",
        "Estos", "Estas", "Ese", "Esa", "Esos", "Esas", "Aquel", "Aquella", "Aquellos",
        "Aquellas", "Mi", "Tu", "Su", "Nuestro", "Nuestra", "Vuestro", "Vuestra", "Sus",
        "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo",
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
        "Septiembre", "Octubre", "Noviembre", "Diciembre"
    }
    segments = re.split(r'([.!?][\s]+)', text)
    capitalized_words = []
    for i, seg in enumerate(segments):
        if re.match(r'[.!?][\s]+', seg):
            continue
        words = re.findall(r'\b[A-Z][\wáéíóúÁÉÍÓÚüÜñÑ]+\b', seg)
        if i == 0 or (i > 0 and re.match(r'[.!?][\s]+', segments[i-1])):
            words = words[1:] if words else []
        capitalized_words.extend(w for w in words if w not in common_words)
    return capitalized_words

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def entities_match(e1, e2):
    if not e1 and not e2:
        return True
    if (e1 and not e2) or (not e1 and e2):
        return False

    max_diff = 0
    if TOLERANCE_LEVEL <= 75: max_diff = 1
    if TOLERANCE_LEVEL <= 50: max_diff = 2
    if TOLERANCE_LEVEL <= 25: max_diff = 3

    if abs(len(e1) - len(e2)) > max_diff:
        return False

    match_threshold = 1.0
    if TOLERANCE_LEVEL <= 75: match_threshold = 0.75
    if TOLERANCE_LEVEL <= 50: match_threshold = 0.5
    if TOLERANCE_LEVEL <= 25: match_threshold = 0.25

    available = e2[:]
    matched = 0
    for word1 in e1:
        for i, word2 in enumerate(available):
            if similarity(word1, word2) >= NAME_SIMILARITY_THRESHOLD:
                del available[i]
                matched += 1
                break

    return matched / len(e1) >= match_threshold if e1 else True

def similar_lengths(l1, l2):
    w1, w2 = len(l1.strip().split()), len(l2.strip().split())
    c1, c2 = len(l1.strip()), len(l2.strip())
    return abs(w1 - w2) <= WORD_TOLERANCE and abs(c1 - c2) <= CHARACTER_TOLERANCE

def numbers_match(n1, n2):
    if not n1 and not n2:
        return True
    if TOLERANCE_LEVEL >= 100:
        return n1 == n2
    if not n1 or not n2:
        return False

    set1, set2 = set(n1.keys()), set(n2.keys())
    common = set1 & set2
    total = set1 | set2
    match_rate = len(common) / len(total) if total else 1.0

    threshold = 1.0
    if TOLERANCE_LEVEL <= 75: threshold = 0.75
    if TOLERANCE_LEVEL <= 50: threshold = 0.5
    if TOLERANCE_LEVEL <= 25: threshold = 0.25

    if match_rate >= threshold:
        for num in common:
            diff = 0
            if TOLERANCE_LEVEL <= 75: diff = 1
            if TOLERANCE_LEVEL <= 50: diff = 2
            if TOLERANCE_LEVEL <= 25: diff = 3
            if abs(n1[num] - n2[num]) > diff:
                return False
        return True
    return False

# Load lines
with open(FILE_IN1, encoding='utf8') as f1:
    lines1 = f1.readlines()

with open(FILE_IN2, encoding='utf8') as f2:
    lines2 = f2.readlines()

stats = {
    "total_lines": min(len(lines1), len(lines2)),
    "length_filter": {"passed": 0},
    "number_filter": {"with_numbers": 0, "without_numbers": 0, "matched": 0},
    "entity_filter": {"with_entities": 0, "without_entities": 0, "matched": 0},
    "final": 0
}

# Step 1: Length filter
filtered1 = []
filtered2 = []
filtered_indices = []
for i in range(stats["total_lines"]):
    if similar_lengths(lines1[i], lines2[i]):
        filtered1.append(lines1[i])
        filtered2.append(lines2[i])
        filtered_indices.append(i)
        stats["length_filter"]["passed"] += 1

# Step 2: Number filter
filtered_n1, filtered_n2, indices_n = [], [], []
for j, i in enumerate(filtered_indices):
    nums1 = extract_and_count_numbers(lines1[i])
    nums2 = extract_and_count_numbers(lines2[i])
    stats["number_filter"]["with_numbers" if nums1 or nums2 else "without_numbers"] += 1
    if not nums1 and not nums2 or numbers_match(nums1, nums2):
        filtered_n1.append(filtered1[j])
        filtered_n2.append(filtered2[j])
        indices_n.append(i)
        stats["number_filter"]["matched"] += 1

# Step 3: Named Entity filter
final1, final2, final_indices, sample_entities = [], [], [], []
for idx, (j, i) in enumerate(enumerate(indices_n)):
    if idx % 50 == 0:
        print(f"Processing named entities: {idx}/{len(indices_n)} lines...")
    ents1 = extract_named_entities(lines1[i])
    ents2 = extract_named_entities(lines2[i])
    if ents1 or ents2:
        sample_entities.append((i, ents1, ents2))
    stats["entity_filter"]["with_entities" if ents1 or ents2 else "without_entities"] += 1
    if entities_match(ents1, ents2):
        final1.append(filtered_n1[j])
        final2.append(filtered_n2[j])
        final_indices.append(i)
        stats["entity_filter"]["matched"] += 1

stats["final"] = len(final1)

FILE_OUT1 = f"{FILE_OUT1.split('.')[0]}_{TOLERANCE_LEVEL}.{FILE_OUT1.split('.')[-1]}"
FILE_OUT2 = f"{FILE_OUT2.split('.')[0]}_{TOLERANCE_LEVEL}.{FILE_OUT2.split('.')[-1]}"

with open(FILE_OUT1, 'w', encoding='utf8') as f:
    f.writelines(final1)
with open(FILE_OUT2, 'w', encoding='utf8') as f:
    f.writelines(final2)

# Report
print(f"\n1. LENGTH FILTER:")
print(f"   - Passed: {stats['length_filter']['passed']} ({stats['length_filter']['passed']/stats['total_lines']*100:.1f}%)")

print(f"\n2. NUMBER FILTER (from {stats['length_filter']['passed']} length-passed lines):")
print(f"   - With numbers: {stats['number_filter']['with_numbers']} ({stats['number_filter']['with_numbers']/stats['length_filter']['passed']*100:.1f}%)")
print(f"   - Without numbers: {stats['number_filter']['without_numbers']} ({stats['number_filter']['without_numbers']/stats['length_filter']['passed']*100:.1f}%)")
print(f"   - Passed: {stats['number_filter']['matched']} ({stats['number_filter']['matched']/stats['length_filter']['passed']*100:.1f}%)")

print(f"\n3. NAMED ENTITY FILTER (from {stats['number_filter']['matched']} number-passed lines):")
print(f"   - With entities: {stats['entity_filter']['with_entities']} ({stats['entity_filter']['with_entities']/stats['number_filter']['matched']*100:.1f}%)")
print(f"   - Without entities: {stats['entity_filter']['without_entities']} ({stats['entity_filter']['without_entities']/stats['number_filter']['matched']*100:.1f}%)")
print(f"   - Passed: {stats['entity_filter']['matched']} ({stats['entity_filter']['matched']/stats['number_filter']['matched']*100:.1f}%)")

print(f"\nTolerance level applied: {TOLERANCE_LEVEL}%")
print(f"Original lines: {stats['total_lines']}")
print(f"Final filtered lines: {stats['final']} ({stats['final']/stats['total_lines']*100:.1f}%)")
