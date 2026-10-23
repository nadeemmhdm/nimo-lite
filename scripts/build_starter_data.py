"""Original authored CC0 examples. No scraping or pretrained-model output is used."""
import json
from pathlib import Path
from nimo.nlp import RESPONSES

QUESTIONS = {
'greeting':['Hello','Hi there','Good morning','Good evening','Hey Nimo','Hello assistant','Hi Nimo Lite','Greetings','Hey there','Good afternoon','Namaskaram','നമസ്കാരം'],
'identity':['Who are you?','What is your name?','Introduce yourself','Tell me about yourself','Are you Nimo?','What assistant is this?','Who am I chatting with?','Please introduce Nimo','Name this assistant','What are you called?','Describe your identity','Ninte peru entha?'],
'help':['What can you do?','How can you help me?','Show available features','What are your capabilities?','Help me get started','What tasks can I try?','How do I use this assistant?','Show me your tools','Give me a quick guide','What can I do here?','Explain your features','I need help with this app'],
'privacy':['Is my data private?','Do you upload my images?','Where does my data go?','Does this work offline?','Are my notes local?','Do you send training data online?','What leaves my computer?','Explain local privacy','Is the internet required?','Can I keep my data offline?','Are there remote model calls?','Tell me about data privacy'],
'teaching':['How do I train you?','How can I teach you?','Start teaching a new concept','How do I add training examples?','How can I correct an answer?','Explain the training steps','Where do I approve examples?','How do I label images?','Help me prepare a dataset','How do I improve your model?','Where is the training center?','I want to teach Nimo'],
'knowledge':['How do I save a note?','What is local knowledge?','How does memory work?','Does saving a fact train weights?','Where can I store facts?','How do I add a sourced document?','Explain knowledge retrieval','Can I search my notes?','How do I remember a fact locally?','What is the knowledge base?','Does memory update the model?','Save facts without retraining'],
'machine_learning':['What is machine learning?','Explain model training','What are model parameters?','Why use held out data?','What is a validation set?','How does a model learn?','Explain learning from examples','What does machine learning fit?','Why separate train and test?','What is generalization?','Explain ML basics','How do weights learn from data?'],
'nlp':['What is natural language processing?','Explain NLP','How do you process language?','What is text classification?','Explain intent recognition','What is a language model?','How does language processing work?','What is the NLP model for?','Explain human language processing','How do you classify a request?','What does NLP mean?','What are text features?'],
'thanks':['Thank you','Thanks Nimo','That helped thanks','Thanks a lot','Thank you for helping','Much appreciated','I appreciate your help','Many thanks','Great thank you','Thanks assistant','Nanni','നന്ദി'],
'uncertainty':['Can you be wrong?','Are your answers always correct?','What if you do not know?','Can I trust every response?','Explain your limitations','Do you know everything?','Will you admit uncertainty?','How should I verify your answers?','Are you a reliable expert?','Can small models make mistakes?','Tell me when you are unsure','Should I check important answers?'],
'other':['What is tomorrow weather?','Who won the football game?','Predict next week stock prices','Book a flight for me','Access my bank account','Execute this shell command','Delete my home directory','Tell me a live news story','What did my friend say?','Diagnose my illness','Solve an unknown research problem','Find my passwords']
}


def main():
    examples=[]
    for label,questions in QUESTIONS.items():
        for question in questions:
            examples.append(dict(question=question,answer=RESPONSES[label],category='intent:'+label,
                source='Nimo Lite contributors: original assistant-basics-v1',license='CC0-1.0'))
    extras=[('What is 2 + 3?','5.','mathematics'),('What is 6 times 7?','42.','mathematics'),
        ('What is 12 divided by 3?','4.','mathematics'),('Python function to add two numbers','def add(a, b):\n    return a + b','coding'),
        ('Reverse a Python string named text','text[::-1]','coding'),('Translate hello to Malayalam','നമസ്കാരം','Malayalam'),
        ('Translate thank you to Malayalam','നന്ദി','Malayalam'),('Manglish greeting example','Namaskaram! Sukham ano?','Manglish')]
    for question,answer,category in extras:
        examples.append(dict(question=question,answer=answer,category=category,
            source='Nimo Lite contributors: original assistant-basics-v1',license='CC0-1.0'))
    out=Path(__file__).resolve().parents[1]/'nimo/resources/assistant-basics.json'
    out.write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(examples)} original examples in {out}')

if __name__=='__main__':main()
