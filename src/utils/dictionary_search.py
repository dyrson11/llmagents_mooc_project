import re
import unicodedata

from rapidfuzz import fuzz
from typing import Dict, Any
from heapq import heappush, heappop

from ..classes.pydantic_model import Palabra, Sufijo, Posposicion


class DictionarySearch:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def fast_similarity(self, a, b):
        return fuzz.ratio(a, b) / 100.0  # Normalize to 0-1

    def smart_similarity(self, candidate, target):
        sub_target = target[: len(candidate)]
        sim_part = fuzz.ratio(candidate, sub_target) / 100.0
        sim_full = fuzz.ratio(candidate, target) / 100.0
        sim = (sim_part + sim_full) / 2.0
        return min(sim, 1.0)

    def find_best_concat(
        self, words: dict, suffixes: dict, target: str, beam_width=5, max_depth=8
    ):
        target = target.lower()
        visited = set()
        heap = []

        # Initialize with words
        for word in words.keys():
            sim = self.smart_similarity(word.lower(), target)
            heappush(heap, (-sim, word, 1, [word]))  # max-heap by negating sim

        results = []

        while heap:
            new_heap = []
            for _ in range(min(len(heap), beam_width)):
                neg_sim, current, depth, path = heappop(heap)
                sim = -neg_sim

                if ((current, depth) in visited) or (depth > max_depth):
                    continue
                visited.add((current, depth))

                results.append((current, sim, path))

                for suffix_key in suffixes.keys():
                    new_str = current + suffix_key
                    new_sim = self.smart_similarity(new_str, target)
                    heappush(
                        new_heap, (-new_sim, new_str, depth + 1, path + [suffix_key])
                    )

            heap = new_heap

        # Sort by similarity and return top matches
        results.sort(key=lambda x: -x[1])
        final_results = []
        for i in range(beam_width):
            word, sim, path = results[i]
            if sim <= 0.5:
                break
            path = [words[path[0]]] + [suffixes[sufijo] for sufijo in path[1:]]
            final_results.append((word, sim, path))
        return final_results

    def get_filtered_characters(sentence):
        result = []
        for char in sentence:
            # Get Unicode category (e.g., "Pd" for dash punctuation, "Po" for other punctuation)
            category = unicodedata.category(char)

            # Allow letters, numbers, spaces, and marks (Mn, Mc, Me), and specific punctuation marks
            if (
                category.startswith(("L", "N", "Z"))  # letters, numbers, separators
                or category.startswith("M")  # marks: accents, dieresis, etc.
                or char in ["´", "`", "^", "¨", "’"]
            ):  # explicitly keep these
                result.append(char)
            # else: remove other punctuation
        return "".join(result)

    def build_prompt(self, word_list: str, possible_translations_sentence: list):
        """Construye un prompt para el modelo."""
        string = ""
        for word, possible_compositions in zip(
            word_list, possible_translations_sentence
        ):
            string += f"\nPosibles composiciones para palabra: {word}\n"
            for i, possible_composition in enumerate(possible_compositions):
                palabra, similitud, partes = possible_composition
                definition_str = ""
                composition_str = []
                for part in partes:
                    if isinstance(part, Palabra):
                        composition_str.append(f"{part.palabra}")
                    elif isinstance(part, Sufijo):
                        composition_str.append(f"-{part.sufijo}")
                    elif isinstance(part, Posposicion):
                        composition_str.append(f"{part.posposicion}")
                    definition_str += f"{part}\n"
                palabra = "".join(composition_str).replace("-", "")
                similitud = self.smart_similarity(palabra, word)
                string += f"\n{i+1}. {palabra}: {' + '.join(composition_str)} (similitud: {similitud})"
                string += f"\n{definition_str}"
        return string

    def process_sentence(
        self, sentence: str, dictionary_to_search: dict, suffixes: dict
    ):
        """
        Pipeline para procesar la oracion y buscar en el diccionario.
        Se siguen los siguientes pasos:
        1. Limpiar espacios
        2. Corregir apostrofes y comillas
        3. Tokenizar
        4. Buscar en el diccionario
        5. Armar prompt
        Se devuelve el prompt generado.
        """
        sentence = re.sub(r"\s+", " ", sentence).strip()
        sentence = self.fix_apostrophes_and_quotes(sentence)
        filtered_sentence = self.get_filtered_characters(sentence).lower()
        word_list = self.tokenize(filtered_sentence)
        possible_translations = []
        for word in word_list:
            translations = self.find_best_concat(dictionary_to_search, suffixes, word)
            possible_translations.append(translations)

        prompt = self.build_prompt(word_list, possible_translations)
        print(prompt)
        return prompt
