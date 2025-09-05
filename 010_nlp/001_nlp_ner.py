import nltk
import numpy #had to install numpy to get nltk to work
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Download required resources (only needed once)
nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Text input
text = "September 2025 - U.S. NewsThe U.S. economy added just 22,000 jobs in August. Tariff uncertainty, immigration policy shifts, and AI adoption are all cited as key factors."

# # Tokenize into words (preserve_line avoids punkt_tab error)
tokens = word_tokenize(text, preserve_line=True)
print(f"\nTokens: {tokens}")

freqDist = FreqDist(tokens)
print(f"\nMost Common tokens:{freqDist.most_common()}")

# Named Entity Recognition
pos_tags = nltk.pos_tag(tokens)
#'NNP' for proper noun, 'VBD' for verb past tense, etc
print(f"\nPOS (Parts of Speech) Tags: {pos_tags}")

#Named Entity Recognition code
ner_tree = nltk.ne_chunk(pos_tags)

print("\nNamed Entities:")
for subtree in ner_tree:
    if hasattr(subtree, 'label'):
        entity = " ".join([token for token, pos in subtree.leaves()])
        print(f"{entity} → {subtree.label()}")

