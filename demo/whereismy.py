import argparse

from message_cortana import DirectLineAPI
code = 'OzNAiUHsamo.q63XOEn6ke-xOdCGt0XJeWVff-O_dWYQcikbfjZzrS8'
bot = DirectLineAPI(code)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', '--object', dest='displaced_obj', type=str, default='bottle', help='Object to locate.')
    args = parser.parse_args()
    msg = "Where is my {0}?".format(args.displaced_obj)
    print(bot.send_message(msg))
    botresponse = bot.get_message()
    print(botresponse)
