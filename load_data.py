import json
import codecs


domain2intents={'banking':['transfer','transactions','balance','freeze account',
'pay bill', 'bill balance', 'bill due', 'interest rate', 'routing number',
'minimum payment', 'order checks', 'pin change', 'report fraud', 'account blocked',
'spending history'],
'credit cards':['credit score', 'report lost card', 'credit limit', 'rewards balance',
'new card', 'application status', 'card declined', 'international fees', 'apr',
'redeem rewards', 'credit limit change', 'damaged card', 'replacement card duration',
'improve credit score', 'expiration date'],
'kitchen & dining':['recipe','restaurant reviews','calories','nutrition info',
'restaurant suggestion', 'ingredients list', 'ingredient substitution', 'cook time',
'food last', 'meal suggestion', 'restaurant reservation', 'confirm reservation',
'how busy', 'cancel reservation','accept reservation'],
'home':['shopping list', 'shopping list update', 'next song', 'play music', 'update playlist',
'todo list', 'todo list update', 'calendar', 'calendar update', 'what song', 'order',
'order status', 'reminder', 'reminder update', 'smart home'],
'auto & commute':['traffic', 'directions', 'gas', 'gas type', 'distance', 'current location',
'mpg', 'oil change when','oil change how', 'jump start', 'uber', 'schedule maintenance',
'last maintenance', 'tire pressure', 'tire change'],
'travel':['book flight', 'book hotel', 'car rental', 'travel suggestion', 'travel alert',
'travel notification', 'carry on', 'timezone', 'vaccines', 'translate', 'flight status',
'international visa', 'lost luggage', 'plug type', 'exchange rate'],
'utility':['time', 'alarm', 'share location', 'find phone', 'weather', 'text', 'spelling', 'make call',
'timer', 'date', 'calculator', 'measurement conversion', 'flip coin', 'roll dice', 'definition'],
'work':['direct deposit', 'pto request', 'taxes', 'payday', 'w2', 'pto balance', 'pto request status',
'next holiday', 'insurance', 'insurance change', 'schedule meeting', 'pto used', 'meeting schedule',
'rollover 401k', 'income'],
'small talk':['greeting', 'goodbye', 'tell joke', 'where are you from', 'how old are you', 'what is your name',
'who made you', 'thank you', 'what can I ask you', 'what are your hobbies', 'do you have pets', 'are you a bot',
'meaning of life', 'who do you work for', 'fun fact'],
'meta':['change AI name', 'change user name', 'cancel', 'user name', 'reset settings', 'whisper mode',
'repeat', 'no', 'yes', 'maybe', 'change language', 'change accent', 'change volume', 'change speed',
'sync device']}

dataIntent_2_realIntent={'routing':'routing number', 'what can i ask you':'what can I ask you', 'change ai name':'change AI name',
'min payment':'minimum payment', 'accept reservations': 'accept reservation'}

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def load_CLINC150_with_specific_domain(domain_name, k):
    gold_intent_set = []
    for domain, intent_list in domain2intents.items():
        gold_intent_set+=intent_list
    gold_intent_set = set(gold_intent_set)

    '''intent to domain'''
    intent2domain={}
    for domain, intent_list in domain2intents.items():
        for intent in intent_list:
             intent2domain[intent] = domain

    readfile = codecs.open('/export/home/Dataset/CLINC150/data_full.json', 'r', 'utf-8')


    interested_intents = domain2intents.get(domain_name)
    assert len(interested_intents) == 15
    file2dict =  json.load(readfile)
    intent_set = set()
    train_intent2examples={}
    dev_intent2examples={}
    test_intent2examples={}

    for key, value in file2dict.items():
        print(key, len(value))
        if key in set(['train', 'val','test']):
            for sub_list in value:
                sentence = sub_list[0].strip()
                intent = ' '.join(sub_list[1].split('_'))
                intent = dataIntent_2_realIntent.get(intent, intent)
                if intent in set(interested_intents):
                    '''this intent is in the target domain'''
                    if key == 'train':
                        examples = train_intent2examples.get(intent)
                        if examples is None:
                            examples = []
                        examples.append(sentence)
                        train_intent2examples[intent] = examples
                    elif key == 'val':
                        examples = dev_intent2examples.get(intent)
                        if examples is None:
                            examples = []
                        examples.append(sentence)
                        dev_intent2examples[intent] = examples
                    else:
                        examples = test_intent2examples.get(intent)
                        if examples is None:
                            examples = []
                        examples.append(sentence)
                        test_intent2examples[intent] = examples
    '''confirm everything is correct'''
    assert len(train_intent2examples.keys()) == 15
    assert len(dev_intent2examples.keys()) == 15
    assert len(test_intent2examples.keys()) == 15
    for key, valuelist in train_intent2examples.items():
        assert len(valuelist) == 100
    for key, valuelist in dev_intent2examples.items():
        assert len(valuelist) == 20
    for key, valuelist in test_intent2examples.items():
        assert len(valuelist) == 30

    '''k-shot sampling'''
    '''train'''
    train_examples = []
    for intent, example_list in train_intent2examples.items():
        sampled_examples = random.sample(example_list, k)
        for example in sampled_examples:
            train_examples.append(
                InputExample(guid='train_ex', text_a=example, text_b=None, label=intent))
    dev_examples = []
    for intent, example_list in dev_intent2examples.items():
        sampled_examples = example_list#random.sample(example_list, k)
        for example in sampled_examples:
            dev_examples.append(
                InputExample(guid='dev_ex', text_a=example, text_b=None, label=intent))
    test_examples = []
    for intent, example_list in test_intent2examples.items():
        sampled_examples = example_list#random.sample(example_list, k)
        for example in sampled_examples:
            test_examples.append(
                InputExample(guid='test_ex', text_a=example, text_b=None, label=intent))

    return train_examples, dev_examples, test_examples, interested_intents



if __name__ == "__main__":
    load_CLINC150()
