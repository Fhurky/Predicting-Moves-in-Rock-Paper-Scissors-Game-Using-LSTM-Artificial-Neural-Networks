# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM
import time

# Veriyi yükleme
data = pd.read_csv('veri.csv')

# Kategorik verileri sayısal değerlere çevirme
label_encoder = LabelEncoder()
data['1.oyuncu'] = label_encoder.fit_transform(data['1.oyuncu'])
data['2.oyuncu'] = label_encoder.transform(data['2.oyuncu'])

# Hamleleri diziler halinde hazırlama
sequence_length = 3  # Önceki 3 hamleye bakarak tahmin yapacağız

def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequence = data['2.oyuncu'].iloc[i:i + sequence_length].values
        target = data['2.oyuncu'].iloc[i + sequence_length]
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)

X, y = create_sequences(data, sequence_length)

# Veriyi one-hot encoding ile dönüştürme
num_classes = len(label_encoder.classes_)
X_encoded = to_categorical(X, num_classes=num_classes)
y_encoded = to_categorical(y, num_classes=num_classes)

# Modeli oluşturma
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, num_classes), return_sequences=True))  # İlk LSTM(Long Short Term Memory) katmanı, return_sequences=True
model.add(LSTM(50))  # İkinci LSTM katmanı
model.add(Dense(50, activation='relu'))  # Yeni bir gizli katman
model.add(Dense(num_classes, activation='softmax'))

# Modeli derleme
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_encoded, y_encoded, epochs=50, batch_size=10, validation_split=0.2)

# 1. oyuncunun hamlesini belirleme fonksiyonu
def determine_move(predicted_move):
    if predicted_move == 0:  # 2. oyuncu taş oynayacaksa
        return 'kağıt'  # 1. oyuncu kağıt oynamalı
    elif predicted_move == 1:  # 2. oyuncu kağıt oynayacaksa
        return 'makas'  # 1. oyuncu makas oynamalı
    else:  # 2. oyuncu makas oynayacaksa
        return 'taş'  # 1. oyuncu taş oynamalı

# 2. oyuncunun hamlesini tahmin etme ve 1. oyuncunun hamlesini belirleme
def play_game(previous_moves):
    # Önceki hamleleri sayısal değere dönüştürme
    previous_moves_encoded = label_encoder.transform(previous_moves)
    previous_moves_encoded = to_categorical(previous_moves_encoded, num_classes=num_classes)
    previous_moves_encoded = np.expand_dims(previous_moves_encoded, axis=0)

    # Bir sonraki hamleyi tahmin etme
    predicted_move = model.predict(previous_moves_encoded)
    predicted_move_class = np.argmax(predicted_move, axis=1)

    # 1. oyuncunun hamlesini belirleme
    next_move = determine_move(predicted_move_class[0])
    return next_move

def predict_next_move_probabilities(previous_moves):
    # Önceki hamleleri sayısal değere dönüştürme
    previous_moves_encoded = label_encoder.transform(previous_moves)
    previous_moves_encoded = to_categorical(previous_moves_encoded, num_classes=num_classes)
    previous_moves_encoded = np.expand_dims(previous_moves_encoded, axis=0)

    # Bir sonraki hamle için olasılıkları tahmin etme
    predicted_probabilities = model.predict(previous_moves_encoded)
    predicted_probabilities = predicted_probabilities[0]  # İlk boyutu kaldır

    # Tahmin edilen olasılıkları taş, kağıt, makas sırasıyla döndürme
    probabilities = {
        'taş': predicted_probabilities[0],
        'kağıt': predicted_probabilities[1],
        'makas': predicted_probabilities[2]
    }
    return probabilities

def get_last_3_moves(data):
    # Son 3 hamleyi 2. oyuncunun yaptığı hamleler olarak al
    return data['2.oyuncu'].tail(3).tolist()

def play_with_timer(model):
    global data

    # Oyun başlatma mesajı
    print("Oyun başlıyor! 3 saniye içinde hamleni yap ve ardından rakibin hamlesini söyle.")

    # 3 saniye bekletme
    time.sleep(3)

    # Veri setindeki son 3 hamleyi al
    last_3_moves = get_last_3_moves(data)
    print(f"2. oyuncunun önceki hamleleri: {last_3_moves}")

    # Kendi hamlesini tahmin etme
    next_move = play_game(last_3_moves)
    print(f"Hamlem: {next_move}")
    
    next_move_int = 0
    
    if next_move == "taş":
        next_move_int = 0
    elif next_move == "kağıt":
        next_move_int = 1
    elif next_move == "makas":
        next_move_int = 2

    # Rakibin hamlesini girmesini isteme
    opponent_move = input("Rakibin hamlesini gir (taş için 0, kağıt için 1, makas için 2): ")

    # Girdiyi kontrol etme ve veri setine ekleme
    if opponent_move in ['0', '1', '2']:
        # Veri setine oyunu ekleme
        new_data = pd.DataFrame({'1.oyuncu': [next_move_int], '2.oyuncu': [opponent_move]})
        data = pd.concat([data, new_data], ignore_index=True)
        print("Oyun veri setine eklendi!")
        
        # Güncellenen veri setini CSV dosyasına kaydetme
        data.to_csv('veri.csv', index=False)
        print("Veri CSV dosyasına kaydedildi!")
    else:
        print("Geçersiz hamle girdisi!")


# Örnek bir oyun oynama
previous_moves = get_last_3_moves(data)
next_move = play_game(previous_moves)
probabilities = predict_next_move_probabilities(previous_moves)

# Tahmin edilen olasılıkları ekrana yazdırma
print(f'2. oyuncunun önceki hamleleri: {previous_moves}')
print('Bir sonraki hamle için tahmin edilen olasılıklar:')
for move, probability in probabilities.items():
    print(f'{move}: %{probability * 100:.2f}')

play_with_timer(model)