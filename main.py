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
import os
import osmnx as ox
import networkx as nx
import matplotlib.image as mpimg

from geopy.distance import geodesic
import matplotlib.pyplot as plt

bot = telebot.TeleBot('token')

G = ox.graph_from_place('Nizhny Novgorod, Russia', network_type='walk')

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

def save_image(image, folder_name, file_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    image_path = os.path.join(folder_name, file_name)
    image.save(image_path)


def calculate_distances(user_location, responses):
    distances = []
    for place, info in responses.items():
        lat, lon = map(float, info['геолокация'].split(', '))
        place_location = (lat, lon)
        distance = geodesic(user_location, place_location).kilometers
        distances.append((place, distance, place_location))
    distances.sort(key=lambda x: x[1])
    return distances

def get_route(origin_point, destination_point):
    try:
        origin_node = ox.distance.nearest_nodes(G, origin_point[1], origin_point[0])
        destination_node = ox.distance.nearest_nodes(G, destination_point[1], destination_point[0])
        shortest_route = nx.shortest_path(G, origin_node, destination_node, weight='length')
        return shortest_route
    except Exception as e:
        print(f"Ошибка при построении маршрута: {e}")
        return []

def get_route_coordinates(route):
    route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
    return route_coords




@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, f'Привет, {message.from_user.first_name}')


@bot.message_handler(commands=['site'])
def site(message):
    webbrowser.open('https://kremlnn.ru/')

user_photos = {}
user_state = {}


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    # Получаем информацию о фотографии
    chat_id = message.chat.id
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    token = 'token'
    file_url = f"https://api.telegram.org/file/bot{token}/{file_info.file_path}"

    # Загружаем изображение из URL
    response = requests.get(file_url)
    image = Image.open(BytesIO(response.content))

    # Выполняем распознавание изображения
    predicted_class = predict_image(image)
    user_state[chat_id] = {'predicted_class': predicted_class, 'image': image, 'file_id': file_id}

        # Отправляем пользователю результат распознавания
    bot.send_message(message.chat.id, f"Это: {predicted_class}")
    if predicted_class in responses:
        response = responses[predicted_class]["описание"]
        photo = responses[predicted_class]["фото"]
        bot.send_message(message.chat.id, response)
        bot.send_photo(message.chat.id, photo)
    else:
        bot.send_message(message.chat.id, "Извините, я не нашел информацию о данной  достопримечательности.")
    bot.send_message(message.chat.id, f"Верно ли я определил достопримечательность? Ответьте 'да' или 'нет'. ")

@bot.message_handler(func=lambda message: message.text.lower() in ['да', 'нет'])
def verify_prediction(message):
    chat_id = message.chat.id
    if message.text.lower() == 'да':
        bot.send_message(message.chat.id, f"Отлично!")
        predicted_class = user_state[chat_id]['predicted_class']
        image = user_state[chat_id]['image']
        save_image(image, predicted_class, f"{user_state[chat_id]['file_id']}.jpg")
    if message.text.lower() == 'нет':

        for idx, (key, value) in enumerate(responses.items()):
            bot.send_message(chat_id, f"{idx + 1}. {key}")
            bot.send_photo(chat_id, value["фото"])
            markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
            for key in responses.keys():
                markup.add(key)
        bot.send_message(chat_id, "Пожалуйста, выберите правильную достопримечательность, нажав на кнопку:", reply_markup=markup)


@bot.message_handler(func=lambda message: message.text in responses)
def handle_correct_name(message):
    chat_id = message.chat.id
    correct_name = message.text
    user_state[chat_id]['corrected_class'] = correct_name
    image = user_state[chat_id]['image']
    save_image(image, correct_name, f"{user_state[chat_id]['file_id']}.jpg")
    if correct_name in responses:
        response = responses[correct_name]["описание"]
        photo = responses[correct_name]["фото"]
        bot.send_message(chat_id, response)
        bot.send_photo(chat_id, photo)
    else:
        bot.send_message(chat_id, "Неверное название. Попробуйте еще раз.")

        # Убираем клавиатуру после выбора
    bot.send_message(chat_id, "Спасибо за ваш выбор!", reply_markup=telebot.types.ReplyKeyboardRemove())

@bot.message_handler(commands=['route'])
def ask_for_location(message):
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    button = telebot.types.KeyboardButton("Отправить местоположение", request_location=True)
    markup.add(button)
    bot.send_message(message.chat.id, "Пожалуйста, отправьте ваше местоположение:", reply_markup=markup)


@bot.message_handler(content_types=['location'])
def handle_location(message):
    if message.location is not None:
        user_location = (message.location.latitude, message.location.longitude)
        distances = calculate_distances(user_location, responses)
        nearest_place = distances[0][0]
        nearest_place_location = distances[0][2]
        route = get_route(user_location, nearest_place_location)
        route_coords = get_route_coordinates(route)

        bot.send_message(message.chat.id, f"Ближайшая достопримечательность: {nearest_place}")

        # Plot the route
        fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='w', show=False, close=False)

        # margin = 0.02  # Add some margin around the route
        # route_nodes = [node for node in route]
        # route_graph = G.subgraph(route_nodes)
        # north, south, east, west = ox.utils_geo.graph_to_gdf(route_graph).total_bounds
        # ax.set_xlim(west - margin, east + margin)
        # ax.set_ylim(south - margin, north + margin)

        # Save the plot as an image file
        image_path = 'route.png'
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

        # Send the image to the user
        with open(image_path, 'rb') as img:
            bot.send_photo(message.chat.id, img)


bot.polling(none_stop=True)
