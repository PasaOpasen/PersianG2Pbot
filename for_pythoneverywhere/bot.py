# - *- coding: utf- 8 - *-
import sys
from importlib import reload
reload(sys)
#sys.setdefaultencoding('utf-8')

import telebot
import os

from PersianG2p import Persian_g2p_converter

Converter = Persian_g2p_converter()
print('-------> Created converter')
tmp = list("آئابتثجحخدذرزسشصضطظعغفقلمنهوپچژکگی")

API_TOKEN = '1222814941:AAF1yiqZZ_CuQfDltC_kqOb_ObSAlp3y0wI'
#API_TOKEN = '1059809966:AAHfjWbOyF3h-F_UZi6krWylOHvA7W3SGE4'

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

    if message.chat.type == 'group':
        if any((r in tmp for r in txt)):
            t = Converter(txt, True, True)
            bot.reply_to(message, t)
    else:
        for r in (r1, r2):
            if txt == r:
                bot.send_message(message.chat.id, db[r])
                return
        bot.send_message(message.chat.id, Converter(txt, True, True))

bot.polling(none_stop=True)