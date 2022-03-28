# EI7007-Pebelejaran-Mesin-Lanjut :technologist:
Tugas terkait dengan pembelajaran mesin lanjut (Deep Learning)

## A step-by-step hyperparameter optimization in convolutional neural network (CNN) & fully connected neural network :scientist:

Pada bagian ini saya akan mencoba melakukan eksplorasi terkait dengan proses optimasi hyperparameter yang terdapat dalam arsitektur CNN dan MLP. 

Pada eksplorasi ini saya akan mencoba menggunakan 2 data dengan 2 jenis persoalan yaitu 
- Persoalan klasifikasi dengan menggunakan data Fashion MNIST
- Persoalan regresi dengan menggunakan data Boston Housing Price

### A. Persoalan Klasifikasi untuk arsitektur CNN (Dataset Fashion MNIST)
Sehingga setelah proses eksplorasi didapatkan informasi terkait dengan hyperparameter pada model CNN yang didapatkan. 

Hyperparameter tersebut seperti:
1. Jumlah convolution layer yang optimal
2. Ukuran filter yang optimal untuk setiap convolution layar
3. Banyaknya filter yang optimal untuk setiap convolution layar
4. Banyaknya hidden unit yang optimal pada bagian fully connected network

Selain hyperparameter di atas, dilakukan eksplorasi terkait dengan **Optimizer**, **learning rate schedule** dan **Losses** pada nilai parameter default untuk mendapatkan kinerja model paling baik

### B. Persoalan Regresi untuk arsitektur MLP (Data Boston Housing Price)

Sehingga setelah proses eksplorasi didapatkan informasi terkait dengan
1. Jumlah hidden layar yang optimal
2. Jumlah hidden unit yang optimal di setiap hidden layar
3. Activation function dengan hasil optimal 
4. Optimizer yang hasil optimal
5. Loss function dengan hasil optimal


## Hasil :man_technologist:
Terdapat 2 hasil eksplorasi dari 2 jenis persoalan, yaitu

1. [Persoalan klasifikasi (CNN)]()

Berdasarkan deskripsi hyperparameter di atas maka berikut deskripsi nilai hyperparamter yang akan dilakukan untuk mendapatkan akurasi model terbaik
- CNN Layers = {2, 3}
- Filter Count inside CNN = {24, 36}
- Kernel Size = {3x3, 5x5}
- Optimizer = {Adam, SGD, RMSprop}
- Learning rate schedule = 0.01
- Losses = {sparse_categorical_crossentropy, categorical_crossentropy}
- Initializer = {GlorotNormal, RandomNormal}
- Dropout = {0.2, 0.7}

Nilai-nilai hyperparameter tersebut berkerja berdasarkan parameter default dari kasus **digit recognition**
- BATCH_SIZE = 64
- EPOCHS = 5

Adapun untuk detail proses ekslporasi dapat dilihat pada tautan berikut. [Tautan eksplorasi]()


2. Persoalan Regresi (MLP)

Berdasarkan deskripsi hyperparameter di atas maka berikut deskripsi nilai hyperparamter yang akan dilakukan untuk mendapatkan akurasi model terbaik

- Jumlah Hidden Layers = {1, 2}
- Hidden Units = {13, 32, 64, 128}
- Activation function = {relu, softmax}
- Optimizer = {Adam, SGD, RMSprop}
- Losses = {Mean Squared Error (MSE)}

Nilai-nilai hyperparameter tersebut berkerja berdasarkan parameter default yaitu
- BATCH_SIZE = 32
- EPOCHS = 1000

Adapun untuk detail proses ekslporasi dapat dilihat pada tautan berikut. [Tautan eksplorasi](https://colab.research.google.com/drive/1JHPNDSkzsralP9g8lWRWW6loKMkyYYVh?usp=sharing)

Dari Hasil proses eksplorasi didapatkan informasi terkait dengan hyperparameter di atas yang menunjukan nilai akurasi terbaik

|No. | N_Hidden_Layer  | N_Hidden_Unit   |Activation_Function    |Optimizer  |Rata-rata Mse Value|
| :--- | :---            |    :----:       |          ---:         |      ---: |              ---: |
|1 |1	|13	|relu	|Adam	|33.93
|2 |1	|13	|relu	|SGD	|6.94
|3 |1	|13	|relu	|RMSprop	|33.64
|4 |1	|13	|relu	|SGD	|6.74
|5 |1	|13	|softmax	|SGD	|12.84
|6 |1	|32	|relu	|SGD	|5.17
|7 |1	|64	|relu	|SGD	|4.85
|8 |1	|128	|relu	|SGD	|3.93
|9 |2	|64 (layer 1)/32 (layer 2)	|relu	|SGD	|3.41
|10 |2	|64 (layer 1)/64 (layer 2)	|relu	|SGD	|2.58
|11 |2	|64 (layer 1)/128 (layer 2)	|relu	|SGD	|2.48
|12 |2	|64 (layer 1)/128 (layer 2)	|softmax	|SGD	|8.88
|13 |2	|128 (layer 1)/32 (layer 2)	|relu	|SGD	|3.46
|14 |2	|128 (layer 1)/64 (layer 2)	|relu	|SGD	|2.15
|15 |2	|128 (layer 1)/128 (layer 2)	|relu	|SGD	|2.24
|18 |2	|128 (layer 1)/128 (layer 2)	|softmax	|SGD	|10.47

Dari hasil eksplorasi didapatkan bahwa kombinasi jumlah hidden layer, hidden unit, activation function dan optimizer yang menghasilkan nilia MSE terkecil adalah pada percobaan eksplorasi No. 14. Berikut baris code model terbaik dengan menggunakan keras python

```python
def build_model():
  model = keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=[len(train_dataset.keys())]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
  ])
  optimizer = tf.keras.optimizers.SGD()
  model.compile(optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mse'])
  return model

model_best = build_model();
model_best.summary()
```

Keluaran Model
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_165 (Dense)           (None, 128)               1792      
                                                                 
 dense_166 (Dense)           (None, 64)                8256      
                                                                 
 dense_167 (Dense)           (None, 1)                 65        
                                                                 
=================================================================
Total params: 10,113
Trainable params: 10,113
Non-trainable params: 0
```

## Kesimpulan
