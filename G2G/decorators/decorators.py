import functools
import time
from typing import Callable, Iterable
import telegram
import logging

with open("../telegram-tokens.txt", mode='r') as telegram_tokens:
    l = telegram_tokens.readlines()
    BOT_TOKEN = l[0].strip()
    CHAT_ID = l[1].strip()

bot = telegram.Bot(token=BOT_TOKEN)
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Formatter:
    def __init__(self, format_function: Callable[[Iterable], str], parse_mode: str = 'Markdown'):
        self.format = format_function
        self.parse_mode = telegram.ParseMode.MARKDOWN if parse_mode == 'Markdown' else telegram.ParseMode.HTML


def logger(formatter: Formatter):
    def logger_decorator(func):
        @functools.wraps(func)
        def logger_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            bot.send_message(chat_id=CHAT_ID, text=formatter.format(result), parse_mode=formatter.parse_mode)
            return result

        return logger_wrapper

    return logger_decorator


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f'Finished in {run_time:.16f} secs')
        return value

    return wrapper_timer
