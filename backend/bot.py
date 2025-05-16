from telegram import Update, InputMediaPhoto
from telegram.ext import CommandHandler, MessageHandler, filters, CallbackContext, Application
import os
from model import PlaceRecognizer
import argparse

recognizer: PlaceRecognizer

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Привет! Отправь мне фото, и я постараюсь понять что это за место.")

async def handle_photo(update: Update, context: CallbackContext):
    photo_file = await update.message.photo[-1].get_file()
    
    file_path = "query_image.jpg"
    await photo_file.download_to_drive(file_path)
    
    label, similar = recognizer.recognize_place(file_path)
    if label:
        message = f"Распознанное место: {label}."

        photos = []
        for i, (img_path, distance) in enumerate(similar, 1):
            photos.append(InputMediaPhoto(media=open(img_path, "rb")))

        await update.message.reply_text(message)
        await update.message.reply_media_group(photos)
    else:
        await update.message.reply_text("Не удалось распознать место.")

def main():
    parser = argparse.ArgumentParser(description='Place recognizer')
    parser.add_argument('-t', '--token', type=str, required=True, help='Telegram Bot Token')
    args = parser.parse_args()

    recognizer = PlaceRecognizer()
    recognizer.load_model()

    application = Application.builder().token(args.token).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    application.run_polling()

if __name__ == "__main__":
    main()