import json
import codecs


domain2intents={'banking':['transfer','transactions','balance','freeze account',
'pay bill', 'bill balance', 'bill due', 'interest rate', 'routine number',
'minimum payment', 'order checks', 'pin change', 'report fraud', 'account blocked',
'spending history'],
'credit cards':['credit score', 'report lost card', 'credit limit', 'rewards balance',
'new card', 'application status', 'card declined', 'international fees', 'apr',
'redeem rewards', 'credit limit change', 'damaged card', 'replacement card duration',
'improve credit score', 'expiration date'],
'kitchen & dining':['recipe','restaurant reviews','calories','nutrition info',
'restaurant suggestion', 'ingredient list', 'ingredient substitution', 'cook time',
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
'meaning of life', 'who do you work from', 'fun fact'],
'meta':['change AI name', 'change user name', 'cancel', 'user name', 'reset settings', 'whispter mode',
'repeat', 'no', 'yes', 'maybe', 'change language', 'change accent', 'change volume', 'change speed',
'sync device']}

def load_CLINC150():
    gold_intent_set = []
    for domain, intent_list in domain2intents.items():
        gold_intent_set+=intent_list
    gold_intent_set = set(gold_intent_set)
    readfile = codecs.open('/export/home/Dataset/CLINC150/data_full.json', 'r', 'utf-8')

    file2dict =  json.load(readfile)
    intent_set = set()
    for key, value in file2dict.items():
        print(key, len(value))
        for sub_list in value:
            sentence = sub_list[0]
            intent = ' '.join(sub_list[1].split('_'))
            intent_set.add(intent.strip())
    print(intent_set-gold_intent_set)





if __name__ == "__main__":
    load_CLINC150()
