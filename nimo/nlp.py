"""Bounded Unicode feature hashing and an explicit, limited assistant response policy.
Intent routing is supervised classification, not language-model generation.
"""
import hashlib
import re
import torch

FEATURES = 2048
RESPONSES = {
    'greeting': "Hello! I'm Nimo Lite. How can I help with your local notes or learning experiments?",
    'identity': "I'm Nimo Lite, your local experimental assistant. My small original models are still learning.",
    'help': 'I can search saved notes, recognize basic assistant requests, and help you prepare and train local datasets.',
    'privacy': 'Your notes, images, and training data stay local by default. Online search sends only the query when you enable it.',
    'teaching': 'Open Training Center, add or import examples, review and approve them, validate the dataset, then start training.',
    'knowledge': 'Save a note with its source in Local Knowledge Base. Retrieval uses the saved text without changing model weights.',
    'machine_learning': 'Machine learning fits model parameters to examples. Separate validation and test data help measure generalization.',
    'nlp': 'Natural language processing works with human language. My intent model classifies requests; my byte model generates text.',
    'thanks': "You're welcome! You can teach me more through reviewed examples and measured training runs.",
    'uncertainty': "I should say when I don't know, show sources for saved knowledge, and avoid claiming untested abilities.",
    'other': "I don't have a reliable answer for that. Try saved knowledge or explicitly choose experimental language-model chat."
}


def features(text, size=FEATURES):
    # Fixed-dimensional features bound memory even for new vocabulary. Hashes are stable across processes.
    normalized = ' '.join(re.findall(r'\w+', text.casefold()))[:4096]
    words = normalized.split()
    parts = ['w:'+w for w in words]
    padded = ' '+normalized+' '
    parts += ['c:'+padded[i:i+3] for i in range(max(0,len(padded)-2))]
    result = torch.zeros(size,dtype=torch.float32)
    for part in parts:
        index = int.from_bytes(hashlib.blake2b(part.encode(),digest_size=8).digest(),'little')%size
        result[index] += 1
    norm = result.norm()
    return result / norm if norm > 0 else result


def intent_label(row):
    return row['content']['category'].removeprefix('intent:')


def assistant_reply(model, checkpoint, text, threshold=.65):
    with torch.inference_mode():
        scores = model(features(text).unsqueeze(0)).softmax(-1)[0]
        values,indices = scores.topk(min(2,len(scores)))
    intent = checkpoint['classes'][int(indices[0])]
    confidence = float(values[0])
    margin = confidence-(float(values[1]) if len(values)>1 else 0)
    # A bounded allowlist decides what can be returned; learned labels cannot select tools or commands.
    accepted = confidence>=threshold and margin>=.2 and intent in RESPONSES
    chosen = intent if accepted else 'other'
    return dict(answer=RESPONSES[chosen],intent=chosen,confidence=confidence,matched=accepted,
                mode='Trained intent routing with authored response; language-model weights not used')
