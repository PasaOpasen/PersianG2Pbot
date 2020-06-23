
import telebot
import os

from flask import Flask, request

from PersianG2p import Persian_g2p_converter

Converter = Persian_g2p_converter()
print('---> Created converter')
tmp = list("آئابتثجحخدذرزسشصضطظعغفقلمنهوپچژکگی")

API_TOKEN = '1222814941:AAG7v8UUXkrjxLQXFaOoxqssHCadpr1mlwM'

bot = telebot.TeleBot(API_TOKEN)

r1 = 'Bot repository'
r2 = 'Algorithm repository'
db = {
    r1: r'https://github.com/PasaOpasen/PersianG2Pbot',
    r2: r'https://github.com/PasaOpasen/PersianG2P'
}

keyboard1 = telebot.types.ReplyKeyboardMarkup(True, True)
keyboard1.row(r1, r2)

instructions = f"""PersianG2P bot converts persian text to phonemes.

Repository of bot: {db[r1]}

Repository of algorithm: {db[r2]}"""


@bot.message_handler(commands=['start', 'help'])
def start_message(message):
    if message.chat.type == 'group':
        bot.send_message(message.chat.id, instructions)
    else:
        bot.send_message(message.chat.id, instructions, reply_markup=keyboard1)


@bot.message_handler(content_types=['text'])
def send_message_global(message):
    txt = message.text

    for r in (r1, r2):
        if txt == r:
            bot.send_message(message.chat.id, db[r])
            return

    if message.chat.type == 'group':
        if any((r in tmp for r in txt)):
            t = Converter(txt, True, True)
            bot.reply_to(message, t)
    else:
        bot.send_message(message.chat.id, Converter(txt, True, True))

bot.polling()


server = Flask(__name__)
PORT = int(os.environ.get('PORT', '8443'))

@server.route('/' + API_TOKEN, methods=['POST'])
def getMessage():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@server.route("/")
def webhook():
    bot.remove_webhook()
    bot.set_webhook(url='https://persiang2p.herokuapp.com/' + API_TOKEN)
    return "!", 200


if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 8443)))
