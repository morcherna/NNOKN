import telebot
import webbrowser
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO



bot = telebot.TeleBot('token')

try:
    with open('database.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        responses = data['database']
except json.JSONDecodeError as e:
    print(f"Ошибка декодирования JSON: {e}")
    responses = {}

def predict_image(image):
    # Преобразование изображения для подачи в модель
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Применение преобразований к изображению
    image = transform(image).unsqueeze(0)

    # Путь к файлу с моделью (замените на ваш путь)
    model_path = 'model_poject.pth'

    # Создание модели и загрузка сохраненных весов
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 15)  # Количество классов равно 15 (замените на ваше количество классов)
    model.load_state_dict(torch.load(model_path))

    # Переключение модели в режим оценки (evaluation mode)
    model.eval()

    # Выполнение предсказания
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    # Получение текстового представления предсказанной метки класса
    class_names = ['Блиновский пассаж', 'Государственный банк',
                   'Кафедральный собор во имя Святого Благоверного Князя Александра Невского',
                   'Нижегородская соборная мечеть', 'Нижегородская ярмарка',
                   'Нижегородский государственный академический театр кукол',
                   'Нижегородский кремль', 'Пакгаузы', 'Памятник гражданину Минину и князю Пожарскому',
                   'Памятник Максиму Горькому', 'Печерский Вознесенский монастырь', 'Речной вокзал',
                   'Усадьба С. М. Рукавишникова',
                   'Храм в честь Собора Пресвятой Богородицы', 'Чкаловская лестница']
    predicted_class = class_names[predicted.item()]

    return predicted_class


@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, f'Привет, {message.from_user.first_name}')


@bot.message_handler(commands=['site'])
def site(message):
    webbrowser.open('https://kremlnn.ru/')

user_photos = {}

# @bot.message_handler(content_types=['photo'])
# def handle_and_send_photo(message):
#     chat_id = message.chat.id
#     if message.chat.id not in user_photos:
#         user_photos[chat_id] = []  # Создаем список для хранения фотографий пользователя, если его еще нет
#     photo_id = message.photo[-1].file_id  # Берем последнюю (самую крупную) фотографию из сообщения
#     user_photos[chat_id].append(photo_id)  # Добавляем идентификатор фотографии в список пользователя
#
#     if user_photos[chat_id]:
#         bot.send_message(chat_id, 'Отправляю ваши фотографии...')
#         for photo_id in user_photos[chat_id]:
#             bot.send_photo(chat_id, photo_id)
#     else:
#         bot.send_message(chat_id, 'У вас еще нет отправленных фотографий.')




@bot.message_handler(content_types=['text'])
def send_response(message):
    query = message.text
    if query in responses:
        response = responses[query]["описание"]
        photo = responses[query]["фото"]
        bot.send_message(message.chat.id, response)
        bot.send_photo(message.chat.id, photo)
    else:
        bot.send_message(message.chat.id, "Извините, я не нашел информацию о данной достопримечательности.")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # Получаем информацию о фотографии
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    bot.token = 'token'
    file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}"

    # Загружаем изображение из URL
    response = requests.get(file_url)
    image = Image.open(BytesIO(response.content))

    # Выполняем распознавание изображения
    predicted_class = predict_image(image)

        # Отправляем пользователю результат распознавания
    bot.send_message(message.chat.id, f"Это: {predicted_class}")
    if predicted_class in responses:
        response = responses[predicted_class]["описание"]
        photo = responses[predicted_class]["фото"]
        bot.send_message(message.chat.id, response)
        bot.send_photo(message.chat.id, photo)
    else:
        bot.send_message(message.chat.id, "Извините, я не нашел информацию о данной  достопримечательности.")

bot.polling(none_stop=True)
