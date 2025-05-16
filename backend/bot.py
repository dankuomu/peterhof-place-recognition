from telegram import Update, InputMediaPhoto
from telegram.ext import CommandHandler, MessageHandler, filters, CallbackContext, Application
import os
from model import PlaceRecognizer
import argparse

recognizer = None

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Привет! Отправь мне фото окрестностей ПУНКа, и я постараюсь понять что это за место.")

async def handle_photo(update: Update, context: CallbackContext):
    global recognizer
    
    status_message = await update.message.reply_text("🔍 Обрабатываю изображение...")
    
    try:
        photo_file = await update.message.photo[-1].get_file()
        
        file_path = "query_image.jpg"
        await photo_file.download_to_drive(file_path)
        
        await status_message.edit_text("🔍 Анализирую изображение...")
        
        label, similar = recognizer.recognize_place(file_path)
        
        if label:
            await status_message.edit_text("✅ Анализ завершен!")
            
            message = f"Распознанное место: {label}."
            await update.message.reply_text(message)
            
            await update.message.reply_text("Вот 3 похожих фотографии:")
            
            photos = []
            for i, (img_path, distance) in enumerate(similar[:3], 1):
                photos.append(InputMediaPhoto(
                    media=open(img_path, "rb")
                ))

            await update.message.reply_media_group(photos)
            
            await status_message.delete()
        else:
            await status_message.edit_text("❌ Не удалось распознать место.")
            
    except Exception as e:
        await status_message.edit_text(f"⚠️ Произошла ошибка: {str(e)}")
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Place recognizer')
    parser.add_argument('-t', '--token', type=str, required=True, help='Telegram Bot Token')
    args = parser.parse_args()

    global recognizer
    recognizer = PlaceRecognizer()
    recognizer.load_model()
    print("Модель успешно загружена")

    application = Application.builder().token(args.token).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    application.run_polling()

if __name__ == "__main__":
    main()