# Laporan Proyek Machine Learning - Nabiel Muaafii Rahman

## Project Overview
Perkembangan teknologi informasi dan pesatnya pertumbuhan data digital telah membawa dampak besar dalam berbagai aspek kehidupan, termasuk industri hiburan. Salah satu bentuk nyata dari pemanfaatan data dalam dunia hiburan adalah sistem rekomendasi film, yang kini menjadi komponen penting dalam meningkatkan pengalaman pengguna di berbagai platform streaming seperti Netflix, Disney+, dan Amazon Prime. Sistem ini membantu pengguna menemukan film yang sesuai dengan preferensi mereka di tengah jumlah pilihan yang sangat banyak.

Dalam membangun sistem rekomendasi yang efektif, dua pendekatan populer yang sering digunakan adalah Content-Based Filtering dan Collaborative Filtering.

Content-Based Filtering bekerja dengan menganalisis karakteristik film yang disukai oleh pengguna, seperti genre, sutradara, aktor, atau deskripsi sinopsis. Sistem ini kemudian merekomendasikan film lain yang memiliki kemiripan atribut dengan film-film yang sebelumnya dinikmati oleh pengguna tersebut. Keunggulan pendekatan ini adalah kemampuannya memberikan rekomendasi yang personal, bahkan untuk pengguna baru yang belum memiliki banyak interaksi.

Di sisi lain, Collaborative Filtering berfokus pada perilaku dan interaksi pengguna. Pendekatan ini mengasumsikan bahwa jika dua pengguna memiliki preferensi yang mirip di masa lalu, maka mereka kemungkinan besar akan menyukai item yang sama di masa depan. Collaborative Filtering terbagi menjadi dua jenis: user-based dan item-based. Meskipun pendekatan ini dapat menghasilkan rekomendasi yang tidak terduga (serendipity), ia juga menghadapi tantangan seperti cold start problem untuk pengguna atau item baru dan sparsity pada data interaksi.

Dengan menggabungkan kedua pendekatan ini, sistem rekomendasi dapat memberikan hasil yang lebih akurat dan relevan. Oleh karena itu, dalam proyek ini dikembangkan sistem rekomendasi film menggunakan pendekatan Content-Based Filtering dan Collaborative Filtering, untuk menganalisis performa masing-masing metode dan mengevaluasi efektivitasnya dalam merekomendasikan film yang sesuai dengan minat pengguna.

## Business Understanding

### Problem Statements
- Bagaimana cara membangun sistem rekomendasi film menggunakan pendekatan Content-Based Filtering dan Collaborative Filtering?
- Sejauh mana efektivitas Content-Based Filtering dalam merekomendasikan film sesuai preferensi pengguna?
- Bagaimana kinerja Collaborative Filtering dalam menangkap pola preferensi pengguna berdasarkan interaksi historis?
- Apa kelebihan dan kekurangan masing-masing pendekatan (Content-Based vs Collaborative Filtering) dalam konteks sistem rekomendasi film?
### Goals
- Membangun sistem rekomendasi film yang mampu memberikan saran film kepada pengguna berdasarkan preferensi mereka, menggunakan dua pendekatan utama: Content-Based Filtering dan Collaborative Filtering.
- Menerapkan metode Content-Based Filtering untuk menghasilkan rekomendasi dengan menganalisis kesamaan atribut antar film berdasarkan genre.
- Menerapkan metode Collaborative Filtering untuk memberikan rekomendasi berdasarkan pola interaksi pengguna, seperti riwayat penilaian.
- Memberikan rekomendasi metode yang paling efektif berdasarkan hasil evaluasi.
### Solution Approach
- Melakukan Exploratory Data Analysis (EDA) untuk mengidentifikasi pola dan tren dalam data film dan penilaian pengguna.
- Menerapkan collaborative filtering menggunakan TensorFlow untuk merekomendasikan destinasi berdasarkan pola penilaian pengguna.
- Menerapkan content-based filtering untuk merekomendasikan destinasi berdasarkan kesamaan atribut genre.

## Data Understanding
Dataset yang diambil pada projek ini berasal dari kaggle https://www.kaggle.com/datasets/nicoletacilibiu/movies-and-ratings-for-recommendation-system?select=movies.csv . Terdapat dua file, file pertama movies.csv berukuran 9742 baris dan berisi tiga fitur movieId, tittle, genres. Dan file kedua ratings.csv dengan ukuran 100836 baris dan empat fitur userId, movieId, rating, dan timestamp.

### Tipe Data
**Data Movie:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9742 entries, 0 to 9741
Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   movieId  9742 non-null   int64 
 1   title    9742 non-null   object
 2   genres   9742 non-null   object
dtypes: int64(1), object(2)
memory usage: 228.5+ KB
```
**Data Rating:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100836 entries, 0 to 100835
Data columns (total 4 columns):
 #   Column     Non-Null Count   Dtype  
---  ------     --------------   -----  
 0   userId     100836 non-null  int64  
 1   movieId    100836 non-null  int64  
 2   rating     100836 non-null  float64
 3   timestamp  100836 non-null  int64  
dtypes: float64(1), int64(3)
memory usage: 3.1 MB
```

### Deskripsi Variabel
Nama Fitur                                    | Deskripsi
----------------------------------------------|------------------------------------------------------------
movieId                                       | Index atau ID dari film
tittle                                        | Judul film
genres                                        | Genre dari dari film
userId                                        | Index atau ID dari user
rating                                        | Rating yang diberikan user pada suatu film
timestamp                                     | Waktu saat user memberikan penilaian



### Melihat Missing Value
**Data Movie:**
```
0
movieId	0
title	0
genres	0

dtype: int64
```
**Data Rating**
```
	0
userId	0
movieId	0
rating	0
timestamp	0

dtype: int64
```
Kedua data pada movie dan rating, bersih dari missing value.

### Melihat duplikasi data
**Data Movie:**
```
np.int64(0)
```
**Data Rating:**
```
np.int64(0)
```
Keduanya tidak terdapat baris duplikat.

### Melihat jumlah user dan Film
```
print('Jumlah userID: ', len(df_rating.userId.unique()))
print('Jumlah movieId: ', len(df_rating.movieId.unique()))
print('Jumlah data rating: ', len(df_rating))
```
Jumlah userID:  610
Jumlah movieId:  9724
Jumlah data rating:  100836

### Visualisasi Data EDA
1. Melihat sebaran statistik pada data rating
   
![image](https://github.com/user-attachments/assets/cc576ab6-0622-4fec-a0b4-642a16219641)

  Berdasarkan hasil deskripsi statistik dataset sistem rekomendasi film yang terdiri dari kolom userId, movieId, rating, dan timestamp, dapat disimpulkan bahwa terdapat sebanyak 100.836 interaksi antara pengguna dan film dalam bentuk pemberian rating. Kolom userId memiliki nilai minimum 1 dan maksimum 610, yang menunjukkan bahwa terdapat 610 pengguna unik dalam dataset.

  Sementara itu, kolom movieId menunjukkan bahwa ID film memiliki nilai minimum 1 dan maksimum 193.609. Untuk kolom rating, nilai rating yang diberikan pengguna berkisar antara 0,5 hingga 5, dengan rata-rata rating sebesar 3,50 dan standar deviasi 1,04. Ini menunjukkan bahwa secara umum pengguna cenderung memberikan rating yang cukup positif. Terakhir, kolom timestamp menunjukkan waktu saat rating diberikan dalam format Unix epoch time, dengan nilai berkisar dari 828.124.600 hingga 1.537.799.000, yang jika dikonversi ke tahun, mencakup rentang waktu dari sekitar tahun 1996 hingga 2018. Data ini sangat berguna untuk membangun sistem rekomendasi berdasarkan perilaku historis pengguna terhadap film tertentu.

2. Bagaimana distribusi genre di film?
   
![image](https://github.com/user-attachments/assets/bb62972d-11b3-498c-8402-5309ead1a20c)

Pada gambar diatas merupakan sebaran atau distribusi genre pada dateset movie. Dapat dilihat genre Drama merupakan genre dengan film terbanyak dengan total 4000 lebih film, dan diakhir terdapat beberapa film yang tidak memiliki genre dengan jumlah kurang dari 1000 film.

## Data Preparation
### Content-Based Filtering
#### Membersihkan text
```
def clean(text):
  text = text.lower()
  text = re.sub(r'\|', ' ', text)
  return text
```
Disiapkan fungsi untuk membersihkan text pada tabel data, tepatnya pada kolom genre. Dalam kode tersbut teks yang dibersihkan hanya diubah kedalam bentuk non-kapital dan menghapus | sebagai pembatas genre.
```
df_movie_clean = df_movie.copy()
df_movie_clean['genres'] = df_movie_clean['genres'].apply(clean)
df_movie_clean.head()
```
Lalu dilakukan pembersihan pada data movie tepatnya kolom genre, sehingga hasil dari pembersihan sebagai berikut:

index|movieId	| title	| genres
------|--------|-------|--------
0|	1|	Toy Story (1995)|	adventure animation children comedy fantasy
1|	2|	Jumanji (1995)|	adventure children fantasy
2|	3|	Grumpier Old Men (1995)|	comedy romance
3|	4|	Waiting to Exhale (1995)|	comedy drama romance
4|	5|	Father of the Bride Part II (1995)|	comedy

#### Melakukan vektorisasi dengan TF-IDF
```
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(df_movie_clean['genres'])
```
TF-IDF (Term Frequency–Inverse Document Frequency) adalah teknik representasi teks yang banyak digunakan dalam pemrosesan bahasa alami (NLP) dan sistem rekomendasi berbasis konten. Tujuannya adalah untuk mengubah data teks menjadi vektor numerik agar bisa diproses oleh algoritma machine learning.

TF (Term Frequency) menghitung seberapa sering sebuah genre muncul dalam satu film, sedangkan IDF (Inverse Document Frequency) mengurangi bobot genre yang terlalu umum di seluruh film. Dengan demikian, genre yang sering muncul dalam sebuah film tapi jarang muncul di film lain akan mendapatkan skor TF-IDF yang tinggi.

TfidfVectorizer() digunakan untuk mengubah kolom 'genres' dari dataframe df_movie_clean yang berisi data teks (seperti "action adventure sci-fi") menjadi matriks TF-IDF. Setiap genre dianggap sebagai sebuah "term", dan setiap film dianggap sebagai "dokumen".

### Collaborative Filtering
#### Menghitung Max dan Min nilai Rating
```
min_rating = df_rating['rating'].min()
max_rating = df_rating['rating'].max()
```
#### Mempersiapkan Data sebelum masuk kedalam model
```
unique_movie_ids = df_rating['movieId'].unique()
movie_to_index = {movie_id: index for index, movie_id in enumerate(unique_movie_ids)}

unique_user_ids = df_rating['userId'].unique()
user_to_index = {user_id: index for index, user_id in enumerate(unique_user_ids)}

# Apply the mapping to the original dataframe before splitting
df_rating['userId_mapped'] = df_rating['userId'].map(user_to_index)
df_rating['movieId_mapped'] = df_rating['movieId'].map(movie_to_index)
```
Untuk mempersiapkan data sebelum digunakan dalam model rekomendasi berbasis Collaborative Filtering, dilakukan proses pemetaan ulang ID pengguna (userId) dan ID film (movieId) ke dalam indeks numerik berturut mulai dari nol. Hal ini penting karena banyak algoritma machine learning, khususnya yang menggunakan layer embedding seperti pada Neural Collaborative Filtering, mensyaratkan input berupa integer yang terurut dari nol hingga N–1. Dalam proses ini, terlebih dahulu diambil seluruh nilai unik dari kolom userId dan movieId, kemudian setiap ID tersebut dipetakan ke angka indeks menggunakan fungsi enumerate. Hasil pemetaan ini disimpan dalam bentuk dictionary (user_to_index dan movie_to_index), yang selanjutnya digunakan untuk menambahkan dua kolom baru ke dalam dataset, yaitu userId_mapped dan movieId_mapped. Kedua kolom ini berisi representasi integer dari ID user dan film, dan akan digunakan sebagai input utama ke dalam model pembelajaran mesin. Dengan demikian, proses ini memastikan bahwa data memiliki format yang sesuai untuk digunakan dalam proses pelatihan model rekomendasi.

#### Memisahkan fitur dan label
```
x = df_rating[['userId_mapped', 'movieId_mapped']] # Use mapped IDs
y = df_rating['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
```
Memisahkan fitur dan label sekaligus menghapus kolom yang tidak dibutuhkan. pada y (label) dilakukan perubahan skala rating secara manual.

#### Splitting Data
```
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
Sebelum data dimasukkan kedalam model, data dibagi menjadi dua bagian, yaitu data latih dan data uji, dengan proporsi 80% untuk data latih dan 20% untuk data uji. Pembagian ini bertujuan agar model dapat melakukan analisis dan pembelajaran tanpa mengalami kebocoran data (data leakage), sehingga hasil evaluasi model menjadi lebih objektif dan dapat dipercaya.

## Modeling
### Content-Based Filtering
```
similarity = cosine_similarity(tfidf)
print(similarity)
```
Digunakan cosine similarity dalam perhitungan kemiripan antar film berdasarkan genre. Cosine similarity adalah sebuah algoritma yang dapat digunakan sebagai metode dalam menghitung tingkat kesamaan (similarity) antar dokumen (Nugroho _et al._ 2021). Perhitungan cosine similarity dilakukan dengan membagi hasil perkalian dot product (perkalian titik) dari dua vektor dengan hasil kali panjang (norma) masing-masing vektor. Semakin kecil sudut antara dua vektor (artinya arah keduanya semakin sejajar), maka nilai cosine similarity-nya mendekati 1, yang menunjukkan tingkat kemiripan yang tinggi. 

### Collaborative Filtering
#### Membangun Model
```
class RecommenderNet(tf.keras.Model):

  # Insialisasi fungsi
  def __init__(self, num_users, num_movie, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_movie = num_movie
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.movie_embedding = layers.Embedding( # layer embeddings movie
        num_movie,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.movie_bias = layers.Embedding(num_movie, 1) # layer embedding movie bias

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    movie_vector = self.movie_embedding(inputs[:, 1]) # memanggil layer embedding 3
    movie_bias = self.movie_bias(inputs[:, 1]) # memanggil layer embedding 4

    dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)

    x = dot_user_movie + user_bias + movie_bias

    return tf.nn.sigmoid(x) # activation sigmoid

early_stopping_cb = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
```
Model di atas merupakan sebuah arsitektur Collaborative Filtering berbasis Neural Network yang dibuat dengan memanfaatkan TensorFlow dan Keras, dan dinamakan RecommenderNet. Model ini dirancang untuk mempelajari representasi laten (embedding) dari pengguna dan item (dalam hal ini movie, yang bisa juga diasumsikan sebagai film), untuk memprediksi seberapa besar kemungkinan seorang pengguna akan menyukai suatu item. Model menggunakan dua buah layer embedding utama: user_embedding dan movie_embedding, yang masing-masing merepresentasikan pengguna dan item dalam bentuk vektor berdimensi rendah (embedding_size). Selain itu, terdapat juga user_bias dan movie_bias untuk menangkap bias individual dari masing-masing pengguna dan item. Dalam fungsi call, model mengalikan (dot product) vektor pengguna dan item untuk menangkap interaksi antara keduanya, lalu menambahkan bias masing-masing. Hasil akhirnya dilewatkan ke fungsi aktivasi sigmoid, sehingga model menghasilkan skor prediksi dalam rentang 0–1, yang bisa diartikan sebagai probabilitas atau minat pengguna terhadap item tersebut. Untuk menghindari overfitting saat pelatihan, digunakan callback EarlyStopping yang akan menghentikan pelatihan jika nilai val_loss tidak membaik selama 5 epoch berturut-turut, sekaligus mengembalikan bobot model ke kondisi terbaik. Model ini cocok digunakan untuk membangun sistem rekomendasi yang bersifat personalisasi tinggi.

#### Melakukan compile model
```
num_users = len(df_rating.userId.unique())
movie_len = len(df_movie)
model = RecommenderNet(num_users, movie_len, 50) # inisialisasi model

# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```
Sebelum dimasukkan ke dalam model, dihitung terlebih dahulu berapa banyak user dan film yang ada dalam database. Lalu model dilakukan compiling dengan perhitungan loss menggunakan BinaryCrossentropy, menggunakan optimizer Adam dengan learning rate 0.001, dan evaluasi metriks menggunakan RMSE.
#### Melakukan fit atau train model
```
history = model.fit(
    x = X_train,
    y = y_train,
    batch_size = 16,
    epochs = 50,
    callbacks = [early_stopping_cb],
    validation_data = (X_test, y_test)
)
```
Terakhir data difitting kedalam model untuk melalui proses training, dengan batch size 16 dan epoch yang digunakan 50. Tak lupa untuk mengaktifkan callback.
**Output:**
```
Epoch 1/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 54s 10ms/step - loss: 0.6504 - root_mean_squared_error: 0.2443 - val_loss: 0.6139 - val_root_mean_squared_error: 0.2092
Epoch 2/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 80s 10ms/step - loss: 0.6099 - root_mean_squared_error: 0.2027 - val_loss: 0.6091 - val_root_mean_squared_error: 0.2039
Epoch 3/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 82s 10ms/step - loss: 0.6039 - root_mean_squared_error: 0.1973 - val_loss: 0.6075 - val_root_mean_squared_error: 0.2019
Epoch 4/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 53s 10ms/step - loss: 0.6000 - root_mean_squared_error: 0.1942 - val_loss: 0.6071 - val_root_mean_squared_error: 0.2015
Epoch 5/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 50s 10ms/step - loss: 0.5986 - root_mean_squared_error: 0.1909 - val_loss: 0.6071 - val_root_mean_squared_error: 0.2012
Epoch 6/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 49s 10ms/step - loss: 0.5980 - root_mean_squared_error: 0.1913 - val_loss: 0.6065 - val_root_mean_squared_error: 0.2003
Epoch 7/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 85s 10ms/step - loss: 0.5982 - root_mean_squared_error: 0.1891 - val_loss: 0.6057 - val_root_mean_squared_error: 0.1994
Epoch 8/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 80s 10ms/step - loss: 0.5966 - root_mean_squared_error: 0.1884 - val_loss: 0.6057 - val_root_mean_squared_error: 0.1993
Epoch 9/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 50s 10ms/step - loss: 0.5957 - root_mean_squared_error: 0.1865 - val_loss: 0.6060 - val_root_mean_squared_error: 0.1996
Epoch 10/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 82s 10ms/step - loss: 0.5954 - root_mean_squared_error: 0.1863 - val_loss: 0.6064 - val_root_mean_squared_error: 0.1999
Epoch 11/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 82s 10ms/step - loss: 0.5942 - root_mean_squared_error: 0.1856 - val_loss: 0.6063 - val_root_mean_squared_error: 0.1998
Epoch 12/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 81s 10ms/step - loss: 0.5936 - root_mean_squared_error: 0.1846 - val_loss: 0.6057 - val_root_mean_squared_error: 0.1990
Epoch 13/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 82s 10ms/step - loss: 0.5946 - root_mean_squared_error: 0.1851 - val_loss: 0.6060 - val_root_mean_squared_error: 0.1995
Epoch 14/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 48s 9ms/step - loss: 0.5938 - root_mean_squared_error: 0.1844 - val_loss: 0.6063 - val_root_mean_squared_error: 0.1997
Epoch 15/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 84s 10ms/step - loss: 0.5942 - root_mean_squared_error: 0.1843 - val_loss: 0.6062 - val_root_mean_squared_error: 0.1993
Epoch 16/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 82s 10ms/step - loss: 0.5920 - root_mean_squared_error: 0.1821 - val_loss: 0.6064 - val_root_mean_squared_error: 0.1996
Epoch 17/50
5042/5042 ━━━━━━━━━━━━━━━━━━━━ 81s 10ms/step - loss: 0.5933 - root_mean_squared_error: 0.1835 - val_loss: 0.6059 - val_root_mean_squared_error: 0.1990
Epoch 17: early stopping
Restoring model weights from the end of the best epoch: 12.
```
### Inference
#### Content-Based Filtering
```
def find_similar_movies(movie_id, num_similar=10):
    # Find the index of the given movie_id
    try:
        movie_index = df_movie_clean[df_movie_clean['movieId'] == movie_id].index[0]
    except IndexError:
        print(f"Movie ID {movie_id} not found in the dataset.")
        return pd.DataFrame()

    # Get the similarity scores for this movie with all other movies
    similarity_scores = list(enumerate(similarity[movie_index]))

    # Sort the movies based on similarity scores in descending order
    sorted_similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the most similar movies (exclude the movie itself)
    top_movie_indices = [i[0] for i in sorted_similarity_scores[1:num_similar+1]]

    # Get the movie details for the similar movies
    similar_movies = df_movie_clean.iloc[top_movie_indices][['title', 'genres']]

    return similar_movies
```
Fungsi find_similar_movies digunakan untuk menghasilkan Top-N rekomendasi film yang mirip dengan film tertentu berdasarkan pendekatan Content-Based Filtering, yang dalam hal ini didasarkan pada kemiripan genre film menggunakan cosine similarity. Fungsi ini pertama-tama mencari indeks dari movieId yang diberikan dalam dataframe df_movie_clean. Setelah menemukan indeks tersebut, fungsi mengambil skor kemiripan (similarity scores) antara film target dengan semua film lainnya dari matriks similarity. Skor-skor ini kemudian diurutkan secara menurun, dan dipilih num_similar film teratas (dengan pengecualian terhadap film itu sendiri). Indeks film-film tersebut digunakan untuk mengambil informasi seperti judul dan genre dari dataframe.

```
# Find similar movies for a movie with movieId 50 (Usual Suspects, The)
movie_id_to_find_similar = 50
similar_movies = find_similar_movies(movie_id_to_find_similar, num_similar=10)

if not similar_movies.empty:
    print(f"\nTop 10 similar movies for Movie ID {movie_id_to_find_similar} ({df_movie_clean[df_movie_clean['movieId'] == movie_id_to_find_similar]['title'].values[0]}):")
similar_movies
```
Contoh penggunaannya adalah dengan memanggil find_similar_movies(50, 10) yang menghasilkan 10 film yang paling mirip dengan film "The Usual Suspects" (Movie ID 50) sebagai berikut:

![image](https://github.com/user-attachments/assets/c0a58a81-e347-4e6b-8584-5e64859be7cd)

#### Collaborative Filtering
```
def recommend_for_user(user_id, num_recommendations=10):
  user_index = user_to_index.get(user_id)
  if user_index is None:
    print(f"User ID {user_id} not found in the training data.")
    return pd.DataFrame()

  movies_not_rated_by_user = df_movie_clean[~df_movie_clean['movieId'].isin(df_rating[df_rating['userId'] == user_id]['movieId'])]['movieId'].values
  movies_not_rated_mapped = [movie_to_index[movie_id] for movie_id in movies_not_rated_by_user if movie_id in movie_to_index]

  if not movies_not_rated_mapped:
      print(f"User ID {user_id} has rated all movies or no new movies are available for recommendation.")
      return pd.DataFrame()

  user_tensor = tf.constant([user_index] * len(movies_not_rated_mapped), dtype=tf.int32)
  movies_tensor = tf.constant(movies_not_rated_mapped, dtype=tf.int32)

  # Stack the user and movie tensors into a single input tensor
  model_input = tf.stack([user_tensor, movies_tensor], axis=1)

  # Make predictions
  predicted_ratings = model.predict(model_input)

  # Get the top recommended movie indices
  top_movie_indices = np.argsort(predicted_ratings.flatten())[::-1][:num_recommendations]

  # Map the recommended movie indices back to original movie IDs
  index_to_movie = {index: movie_id for movie_id, index in movie_to_index.items()}
  recommended_movie_ids = [index_to_movie[movies_not_rated_mapped[i]] for i in top_movie_indices]

  # Get the movie details for the recommended movies
  recommended_movies = df_movie_clean[df_movie_clean['movieId'].isin(recommended_movie_ids)]

  return recommended_movies[['title', 'genres']]
```
Fungsi recommend_for_user dirancang untuk menghasilkan Top-N rekomendasi film bagi pengguna tertentu menggunakan pendekatan Collaborative Filtering berbasis model neural network (RecommenderNet). Fungsi ini bekerja dengan terlebih dahulu mencari indeks dari user_id yang diberikan menggunakan mapping user_to_index. Selanjutnya, sistem mengidentifikasi daftar film yang belum pernah diberi rating oleh pengguna tersebut, kemudian memetakannya ke dalam indeks movie_to_index agar sesuai dengan format input model. Kemudian, model membentuk tensor berisi kombinasi pasangan (user, movie) untuk semua film yang belum diberi rating, dan melakukan prediksi skor preferensi untuk setiap film tersebut menggunakan model RecommenderNet. Skor-skor tersebut diurutkan secara menurun, dan num_recommendations film dengan skor tertinggi akan diambil sebagai rekomendasi.
```
# Example usage:
user_id_to_recommend_for = 1 # Replace with the desired user ID
recommendations = recommend_for_user(user_id_to_recommend_for, num_recommendations=10)

if not recommendations.empty:
    print(f"Top 10 movie recommendations for User ID {user_id_to_recommend_for}:")
recommendations
```
Output dari fungsi ini berupa daftar judul film beserta genre-nya yang direkomendasikan untuk user yang dimaksud, seperti dalam contoh: rekomendasi untuk user_id = 1.

![image](https://github.com/user-attachments/assets/d45283b3-25c9-48a8-b32a-0e67907827a6)

## Evaluation

### Penjelasan Matriks
Matriks evaluasi yang digunakan antara lain **akurasi, precision, recall, dan F1-score**, tetapi pada projek ini lebih difokuskan pada akurasi dan F1-score.

#### 1. Precission@k
Digunakan untuk mengevaluasi seberapa banyak rekomendasi yang benar-benar relevan dari total 𝐾 item yang direkomendasikan oleh sistem.

![image](https://github.com/user-attachments/assets/1eefe4d5-3a19-4304-b731-21c144ed4f5d)

#### 2. Root Mean Squared Error
RMSE adalah ukuran seberapa besar kesalahan rata-rata prediksi model terhadap nilai aktual dalam satuan yang sama dengan target.

![image](https://github.com/user-attachments/assets/e422384c-7e93-4192-947a-b70e0c2e2ca6)

keterangan:
![image](https://github.com/user-attachments/assets/e9583fef-862a-4f82-b2ed-1cce0434bfd3)


### Hasil Evaluasi
#### Content-Based dengan Recall@5
```
def precision_at_k(similarity, df_movie_clean, k=5, num_eval_movies=100):
    num_movies = similarity.shape[0]
    random_movie_indices = np.random.choice(num_movies, num_eval_movies, replace=False)
    total_precision = 0

    genre_sets = [
        set(genres.split()) if pd.notnull(genres) else set()
        for genres in df_movie_clean['genres']
    ]

    for target_idx in random_movie_indices:
        similarity_scores = similarity[target_idx]
        sorted_indices = np.argsort(similarity_scores)[::-1]
        # Exclude the target movie itself
        recommended_indices = [i for i in sorted_indices if i != target_idx][:k]

        if not recommended_indices:
            continue  # Skip if no recommendations are made

        target_genres = genre_sets[target_idx]

        # Define "relevant" recommendations based on shared genres
        relevant_recommended_count = 0
        for rec_idx in recommended_indices:
            if len(target_genres.intersection(genre_sets[rec_idx])) > 0:
                relevant_recommended_count += 1

        total_precision += relevant_recommended_count / k

    return total_precision / num_eval_movies if num_eval_movies > 0 else 0

precision_value = precision_at_k(similarity, df_movie_clean, k=5, num_eval_movies=100)
print(f"Precision@5: {precision_value}")
```
**Output:**
```
Precision@5: 1.0
```
Nilai Precision@5 sebesar 1.0 menunjukkan bahwa seluruh rekomendasi film yang diberikan oleh model dalam lima peringkat teratas adalah relevan dengan film target berdasarkan kemiripan genre. Artinya, setiap film yang direkomendasikan memiliki setidaknya satu genre yang sama dengan film yang menjadi acuan. Hal ini mencerminkan bahwa model content-based filtering yang dibangun mampu secara akurat memilih item-item yang sesuai secara konten, khususnya dalam hal genre. Meskipun hasil ini terlihat sangat ideal, penting untuk dicatat bahwa precision yang tinggi belum tentu menjamin kualitas rekomendasi secara keseluruhan.

#### Collaborative Filtering

![image](https://github.com/user-attachments/assets/eb6e307f-d250-4567-991f-0811d7cab0ef)

Grafik di atas menunjukkan performa model rekomendasi selama proses pelatihan berdasarkan metrik Root Mean Squared Error (RMSE) terhadap data pelatihan (train) dan data pengujian (test) pada setiap epoch. Terlihat bahwa nilai RMSE pada data pelatihan terus menurun secara konsisten hingga akhir pelatihan, menandakan bahwa model semakin baik dalam mempelajari pola dari data latih. Namun, nilai RMSE pada data pengujian menurun di awal, tetapi kemudian cenderung stagnan dan mulai menunjukkan kecenderungan datar setelah sekitar epoch ke-5 hingga ke-16. Pola ini menunjukkan indikasi awal dari overfitting, di mana model terlalu menyesuaikan diri dengan data pelatihan dan tidak lagi mengalami peningkatan performa terhadap data baru. Oleh karena itu, model sebaiknya dievaluasi untuk menentukan titik optimal pelatihan (early stopping), agar dapat menghasilkan performa yang baik dan generalisasi yang lebih kuat.

## Kesimpulan
Proyek ini berhasil membangun sistem rekomendasi film menggunakan dua pendekatan utama, yaitu Content-Based Filtering dan Collaborative Filtering. Melalui pendekatan Content-Based Filtering, sistem menganalisis kemiripan antar film berdasarkan genre menggunakan metode TF-IDF dan cosine similarity, menghasilkan rekomendasi yang relevan secara konten. Sementara itu, pendekatan Collaborative Filtering dibangun menggunakan model deep learning berbasis embedding untuk mempelajari pola interaksi pengguna dengan film berdasarkan riwayat rating. Evaluasi model dilakukan dengan berbagai metrik, seperti RMSE untuk model collaborative dan Precision@5 untuk model content-based. Hasil evaluasi menunjukkan bahwa model content-based mampu mencapai Precision@5 sebesar 1.0, yang berarti seluruh rekomendasi berada dalam kategori relevan berdasarkan genre. Sedangkan model collaborative menunjukkan penurunan error yang stabil sepanjang epoch, namun mulai mengalami gejala overfitting setelah beberapa iterasi. Dengan demikian, kedua pendekatan memiliki kekuatan dan kelemahannya masing-masing. Model content-based unggul dalam akurasi genre, sedangkan model collaborative lebih baik dalam memahami pola perilaku pengguna. Kombinasi keduanya berpotensi membentuk sistem rekomendasi yang lebih kuat dan personal.

## Saran
1. Lakukan hybrid filtering untuk menggabungkan kelebihan Content-Based dan Collaborative Filtering dalam satu sistem rekomendasi.
2. Tambahkan fitur metadata lain, seperti deskripsi film, sutradara, atau aktor, agar model content-based menghasilkan rekomendasi yang lebih kontekstual.
3. Evaluasi dengan lebih banyak metrik, seperti F1-score, MAP@K, atau NDCG, untuk menggambarkan performa sistem secara menyeluruh.
   
## Refrensi
Kaggle. (2025). Movies & Ratings for Recommendation System. Diakses pada 16 Juni 2025 dari
https://www.kaggle.com/datasets/nicoletacilibiu/movies-and-ratings-for-recommendation-system?select=ratings.csv

Nugroho, F. A., Septian, F., Pungkastyo, D. A., & Riyanto, J. (2021). Penerapan algoritma cosine similarity untuk deteksi kesamaan konten pada sistem informasi penelitian dan pengabdian kepada masyarakat. Jurnal Informatika Universitas Pamulang, 5(4), 529-536.

Dicoding. (2024). Machine Learning Terapan. Diakses pada 25 Mei 2025 dari https://www.dicoding.com/academies/319-machine-learning-terapan.
