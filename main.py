import os
import numpy as np
import cv2
from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from deep_translator import GoogleTranslator

load_dotenv()
BOT_TOKEN: str = os.getenv("BOT_TOKEN_")
model = ResNet50()
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я бот, который поможет тебе по фото классифицировать предмет.")

@dp.message_handler(commands=['help'])
async def send_welcome(message: types.Message):
    await message.reply("отправь фото и я расскажу, что я на нем вижу!")

@dp.message_handler(content_types=types.ContentType.PHOTO)
async def save_photo(message: types.Message):
    photo = message.photo[-1]
    photo_id = photo.file_id
    photo_path = await bot.get_file(photo_id)

    file_name = f"img.jpg"
    await photo_path.download(file_name)
    img = cv2.imread("img.jpg")
    resized_img = cv2.resize(img, (224, 224))
    resized_img = preprocess_input(np.reshape(resized_img, (1, 224, 224, 3)))
    els = decode_predictions(model.predict(resized_img))
    out_arr = []
    for el in els:
        for elik in el[:3]:
            out_arr.append(elik[1])
    out_ru = ""
    for els in out_arr:
        translation = GoogleTranslator(source='en', target='ru').translate(els)
        out_ru += translation + ", "
    await message.reply(out_ru[:len(out_ru)-2])

@dp.message_handler()
async def echo(message: types.Message):
    await message.reply("К сожалению. Я умею работать только с картинками. Отправь мне картинку и узнай, что я на ней вижу.")


if __name__ == '__main__':
    executor.start_polling(dp)
