# EI7007-Pebelejaran-Mesin-Lanjut :technologist:
Tugas terkait dengan pembelajaran mesin lanjut (Deep Learning)

## A step-by-step hyperparameter optimization in convolutional neural network (CNN) & Fully Connected Neural Network (MLP) :scientist:

Pada bagian ini saya akan mencoba melakukan eksplorasi terkait dengan proses optimasi hyperparameter yang terdapat dalam arsitektur CNN dan MLP. 

Pada eksplorasi ini saya akan mencoba menggunakan 2 data dengan 2 jenis persoalan yaitu 
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


## Result :man_technologist:
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


2. [Persoalan Regresi (MLP)]()

Berdasarkan deskripsi hyperparameter di atas maka berikut deskripsi nilai hyperparamter yang akan dilakukan untuk mendapatkan akurasi model terbaik

- Hidden Layers = {2, 3, 4}
- Hidden Units = {32, 64, 128}
- Activaton function = {relu, sigmoid, softmax, softsign, tanh, selu, elu, exponential}
- Optimizer = {Adam, SGD, RMSprop}
- Losses = {sparse_categorical_crossentropy, categorical_crossentropy}

Nilai-nilai hyperparameter tersebut berkerja berdasarkan parameter default dari kasus **digit recognition**
- BATCH_SIZE = 64
- EPOCHS = 5

## Conclution
