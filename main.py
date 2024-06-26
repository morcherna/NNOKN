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
import folium
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image


from geopy.distance import geodesic


bot = telebot.TeleBot('token')


place = 'Nizhny Novgorod, Russia'
graph = ox.graph_from_place(place, network_type='walk')

try:
    with open('database.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        responses = data['database']
except json.JSONDecodeError as e:
    print(f"Ошибка декодирования JSON: {e}")
    responses = {}

def predict_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    model_path = 'model_poject.pth'
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 15)
    model.load_state_dict(torch.load(model_path))


    model.eval()


    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)


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


def find_nearest_places(user_location, count=1):
    distances = []
    for place, info in responses.items():
        lat, lon = map(float, info['геолокация'].split(', '))
        place_location = (lat, lon)
        distance = geodesic(user_location, place_location).kilometers
        distances.append((place, distance, place_location))
    distances.sort(key=lambda x: x[1])
    return distances[:count]

def plot_route_to_places(user_location, nearest_places):
    orig_node = ox.distance.nearest_nodes(graph, user_location[1], user_location[0])
    map = folium.Map(location=user_location, zoom_start=13)
    folium.Marker(location=user_location, popup="Ваше местоположение", icon=folium.Icon(color='blue')).add_to(map)

    min_lat, min_lon = user_location
    max_lat, max_lon = user_location

    for place, _, place_location in nearest_places:
        dest_node = ox.distance.nearest_nodes(graph, place_location[1], place_location[0])
        route = nx.shortest_path(graph, orig_node, dest_node, weight='length')
        route_map = ox.plot_route_folium(graph, route, route_map=map, tiles='openstreetmap')
        folium.Marker(location=place_location, popup=place, icon=folium.Icon(color='red')).add_to(map)
        orig_node = dest_node

        min_lat = min(min_lat, place_location[0])
        min_lon = min(min_lon, place_location[1])
        max_lat = max(max_lat, place_location[0])
        max_lon = max(max_lon, place_location[1])

    map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    map_html_path = 'route_map.html'
    map.save(map_html_path)
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=service, options=options)
    driver.get('file://' + os.path.abspath(map_html_path))
    time.sleep(2)
    route_image_path = 'route.png'
    driver.save_screenshot(route_image_path)
    driver.quit()
    image = Image.open(route_image_path)
    cropped_image = image.crop((0, 0, image.width, image.height))
    cropped_image.save(route_image_path)
    return route_image_path

@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, f'Привет, {message.from_user.first_name}')


user_photos = {}
user_state = {}

@bot.message_handler(commands=['photo'])
def ask_for_location(message):
    bot.send_message(message.chat.id, "Отправьте мне фотографию достопримечателности")



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


    bot.send_message(chat_id, "Спасибо за ваш выбор!", reply_markup=telebot.types.ReplyKeyboardRemove())

@bot.message_handler(commands=['route'])
def ask_for_location(message):
    bot.send_message(message.chat.id, "Настраиваюсь на работу...")
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    button = telebot.types.KeyboardButton("Отправить местоположение", request_location=True)
    markup.add(button)
    bot.send_message(message.chat.id, "Пожалуйста, отправьте ваше местоположение:", reply_markup=markup)


@bot.message_handler(content_types=['location'])
def handle_location(message):
    chat_id = message.chat.id
    user_location = (message.location.latitude, message.location.longitude)
    user_state[chat_id] = {'location': user_location}

    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    markup.add("До одной ближайшей", "До двух ближайших", "До трех ближайших")
    bot.send_message(chat_id, "Сколько ближайших местоположений вы хотите увидеть?", reply_markup=markup)



@bot.message_handler(
    func=lambda message: message.text in ["До одной ближайшей", "До двух ближайших", "До трех ближайших"])
def handle_route_request(message):
    bot.send_message(message.chat.id, "Уже строю маршрут...")
    chat_id = message.chat.id
    user_location = user_state[chat_id]['location']

    if message.text == "До одной ближайшей":
        count = 1
    elif message.text == "До двух ближайших":
        count = 2
    elif message.text == "До трех ближайших":
        count = 3

    nearest_places = find_nearest_places(user_location, count=count)
    route_image_path = plot_route_to_places(user_location, nearest_places)

    # Собрать названия ближайших мест
    places_names = [place for place, _, _ in nearest_places]
    places_message = "Маршрут построен до следующих достопримечательностей:\n" + "\n".join(places_names)

    # Отправить названия ближайших мест
    bot.send_message(chat_id, places_message)

    # Отправить изображение маршрута
    with open(route_image_path, 'rb') as img:
        bot.send_photo(chat_id, img)

bot.polling(none_stop=True)
