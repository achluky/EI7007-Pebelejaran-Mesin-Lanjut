# EI7007-Pebelejaran-Mesin-Lanjut :technologist:
Tugas terkait dengan pembelajaran mesin lanjut (Deep Learning)

Nama : Ahmad Luky Ramdani

NIM: 33221020
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
