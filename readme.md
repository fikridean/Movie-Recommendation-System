# Laporan Proyek Machine Learning - Fikri Dean Radityo

## Project Overview
Platform *streaming online*, atau yang dikenal sebagai layanan *streaming*, merupakan penyedia hiburan seperti musik, film, dan serial televisi yang menyampaikan konten melalui internet langsung ke perangkat pengguna. Layanan ini memungkinkan pengguna mengakses konten multimedia secara instan tanpa perlu diunduh, baik secara online maupun offline (Muhd Shukri et al., 2024). Dengan perkembangan pesat teknologi digital, platform *streaming online* mendapatkan perhatian luas dan menjadi bagian penting dalam kehidupan modern. Perubahan budaya global yang beralih ke konsumsi konten digital semakin memperkuat posisi pasar platform *streaming* sebagai media utama hiburan (Sekartaji, 2023).

Setiap platform media *streaming* menyediakan ribuan film dan serial. Sistem rekomendasi menjadi elemen penting yang membantu mempersonalisasi pengalaman pengguna dan memudahkan mereka menemukan konten yang relevan. Dari sudut pandang bisnis, konten yang lebih relevan meningkatkan keterlibatan pengguna, yang pada akhirnya berkontribusi pada pendapatan platform. Sekitar 35 hingga 40% pendapatan platform *streaming* berasal dari sistem rekomendasi (Muvi.com). Penelitian menunjukkan bahwa platform yang menggunakan sistem rekomendasi lebih efektif dalam mempertahankan pengguna dibandingkan yang tidak menggunakan fitur tersebut (Muhd Shukri et al., 2024).

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi yang lebih cerdas dan relevan, sehingga dapat meningkatkan kepuasan pengguna sekaligus membantu platform *streaming* dalam mencapai keberlanjutan bisnisnya.

## Business Understanding
Masalah utama yang dihadapi oleh platform *streaming online* adalah pengembangan sistem rekomendasi yang lebih efektif. Pengguna sering mencari film dengan genre yang sama seperti yang mereka sukai atau tonton sebelumnya, tetapi mereka kesulitan menemukan film yang relevan. Selain itu, banyak pengguna ingin direkomendasikan film serupa setelah menonton film tertentu tanpa harus mencarinya. Konten yang relevan dapat meningkatkan kepuasan dan keterlibatan pengguna, yang berdampak pada pendapatan platform, karena sekitar 35-40% pendapatan berasal dari rekomendasi. Proyek ini bertujuan untuk menciptakan sistem rekomendasi yang lebih tepat dan efisien untuk meningkatkan pengalaman pengguna dan mendukung keberlanjutan bisnis platform *streaming*.

### Problem Statements
- Bagaimana cara memahami dan memperoleh informasi mengenai data yang digunakan dalam pembuatan model sistem rekomendasi?
- Bagaimana cara membangun model sistem rekomendasi dengan menggunakan pendekatan *content-based filtering*?
- Bagaimana cara mengembangkan model sistem rekomendasi dengan pendekatan *collaborative filtering*?
- Bagaimana cara menilai kinerja model sistem rekomendasi yang telah dikembangkan?

### Goals
- Melakukan eksplorasi data awal (EDA) dan visualisasi untuk memahami *dataset* terlebih dahulu.
- Membangun sistem rekomendasi film menggunakan pendekatan *content-based filtering*.
- Membangun sistem rekomendasi film menggunakan pendekatan *collaborative filtering*.
- Mengevaluasi kinerja model sistem rekomendasi yang telah dibuat.

### Solution Approach
Untuk memecahkan rumusan masalah dengan mencapai *goals* yang ditentukan, dibutuhkan pendekatan solusi yang dapat membantu meraih *goals* seperti:
- Melakukan eksplorasi terhadap *dataset* dengan melakukan *exploratory data analysis* untuk mendapatkan informasi-informasi seperti panjang baris dan kolom dan tipe data dari masing-masing kolom, serta mendapatkan visualisasi dengan menampilkan *graph*.
- Membangun sistem rekomendasi menggunakan pendekatan *content-based filtering* dengan melakukan proses *data cleaning* yang dimana termasuk proses *removal duplicates and NaN data*, *removal irrelevant columns*, *handle imbalance data*, dan *text processing*. Selain itu, proses *data transformation*, *calculating cosine similarity*, dan *data transformation* juga akan diterapkan dalam pengembangan sistem rekomendasi ini. Terakhir, akan dilakukan pembuatan fungsi model yang diikuti dengan percobaan prediksi pada model yang telah dibuat.
- Membangun sistem rekomendasi menggunakan pendekatan *collaborative filtering* dengan melakukan proses *data cleaning* yang dimana termasuk proses *removal duplicates and NaN data*, *removal irrelevant columns*, *handle imbalance data*, dan *text processing*. Selain itu, proses *data encoding*, *data transformation*, dan *train test split* juga akan diterapkan dalam pengembangan sistem rekomendasi ini. Terakhir, akan dilakukan pembuatan fungsi model yang dimana model akan dilatih yang kemudian dilakukan percobaan prediksi pada model yang telah dibuat.
- Mengevaluasi kinerja model sistem rekomendasi yang telah dibuat dengan menggunakan metrik evaluasi *Precision* untuk model yang menggunakan teknik *content based filtering* dan metrik evaluasi *Root Mean Squared Error* (RMSE) untuk model yang menggunakan teknik *collaborative filtering*.

## Data Understanding
### *Import Dataset*
Berikut dataset yang digunakan pada proyek ini

- ***Dataset source***: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data
- ***Dataset name***: MovieLens 20M 

*Dataset* yang digunakan untuk pembangunan model machine learning ini adalah *dataset* "MovieLens 20M Dataset *dataset*" yang tersedia di situs web Kaggle pada [tautan](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data) ini. *Dataset* tersebut adalah *dataset* yang terdiri atas beberapa berkas antara lain:

- `tag.csv`
- `rating.csv`
- `movie.csv`
- `link.csv`
- `genome_scores.csv`
- `genome_tags.csv`

Dari berkas-berkas tersebut, berkas yang cocok dan akan digunakan pada proyek ini untuk membuat model dengan teknik content based filtering dan collaborative filtering adalah berkas `movie.csv` dan `rating.csv`.

Berikut ini adalah informasi lainnya mengenai *dataset* tersebut:

Variabel-variabel pada berkas `movie.csv` adalah sebagai berikut:
- `movieId`: *Identifier* yang digunakan untuk film
- `title`: Judul dari film yang ada pada berkas
- `genres`: Genre dari tiap film yang ada pada berkas

Variabel-variabel pada berkas `rating.csv` adalah sebagai berikut:
- `userId`: *Identifier* yang digunakan untuk user yang memberikan rating pada film
- `movieId`: *Identifier* yang digunakan untuk film
- `rating`: *Rating* yang diberikan oleh user untuk film
- `timestamp`: Waktu data berhasil dimasukkan ke dalam berkas

Mengambil kode tautan *google drive file sharing* yang akan digunakan untuk *download* *dataset* dan memasukkannya ke lokal.

```python
dataset_file_id = "1dU0Oazz0S8dT-Y3bxQoJ2D-V4rFUMAkh"

destination = "/content/movie.csv"

gdown.download(f"https://drive.google.com/uc?id={dataset_file_id}", destination, quiet=False)
```

```python
dataset_file_id = "19z9kWCobcD72Z2vI2gi6Cpx92_M5YZnL"

destination = "/content/rating.csv"

gdown.download(f"https://drive.google.com/uc?id={dataset_file_id}", destination, quiet=False)
```

Membuat *pandas dataframe* dari `movie.csv` dan `rating.csv` dan memasukkannya ke variabel.

```python
dataset_movie = "movie.csv"
dataset_rating = "rating.csv"

movie_df = pd.read_csv(dataset_movie)
rating_df = pd.read_csv(dataset_rating)
```

### *Exploratory Data Analysis*
> ***Exploratory Data Analysis*** (EDA) adalah tahap awal dalam mengeksplorasi data untuk menganalisis karakteristik, mengidentifikasi pola, menemukan anomali, dan memverifikasi asumsi-asumsi yang ada dalam data.

Mengambil panjang baris dan kolom serta nama-nama kolom yang ada di `movie_df`.

```python
print('movie_df row and column length:', movie_df.shape)
print('movie_df columns: ', movie_df.keys())
```

Kode di atas menghasilkan *output* berikut:

```python
movie_df row and column length: (27278, 3)
movie_df columns:  Index(['movieId', 'title', 'genres'], dtype='object')
```

Berdasarkan *output* kode di atas didapatkan informasi mengenai `movie_df` sebagai berikut:

*   `movie_df` berisi jumlah baris dan kolom adalah 27278 baris dan 3 kolom.
*   Terdapat 3 kolom pada `movie_df` yaitu `movieId`, `title`, `genres`.

Mengambil informasi dari kolom-kolom yang ada di `movie_df`.

```python
print('movie_df columns information')
movie_df.info()
```

Kode di atas menghasilkan *output* berikut:

```python
movie_df columns information
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 27278 entries, 0 to 27277
Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   movieId  27278 non-null  int64 
 1   title    27278 non-null  object
 2   genres   27278 non-null  object
dtypes: int64(1), object(2)
memory usage: 639.5+ KB
```

Berdasarkan *output* kode di atas didapatkan informasi berikut:

*   Terdapat 2 kolom dengan tipe object, yaitu: `title` dan `genres`. Kolom ini merupakan categorical features (fitur non-numerik).
*   Terdapat 1 kolom numerik dengan tipe data int64, yaitu: `movieId`.

Mengambil informasi statistik dari `movie_df`.

```python
print('movie_df statistic information')
movie_df.describe()
```

Kode di atas menghasilkan *output* berikut:

```python
movie_df statistic information
```

| Statistic | Value         |
|-----------|---------------|
| Count     | 27,278.000    |
| Mean      | 59,855.481    |
| Std       | 44,429.315    |
| Min       | 1.000         |
| 25%       | 6,931.250     |
| 50%       | 68,068.000    |
| 75%       | 100,293.250   |
| Max       | 131,262.000   |

Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:
*   **Count**  adalah jumlah sampel pada data.
*   **Mean** adalah nilai rata-rata.
*   **Std** adalah standar deviasi.
*   **Min** yaitu nilai minimum setiap kolom.
*   **25%** adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
*   **50%** adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
*   **75%** adalah kuartil ketiga.
*   **Max** adalah nilai maksimum.

Melihat contoh data dari `movie_df` dengan mengambil 5 baris pertama.

```python
print('movie_df top 5 rows example')
movie_df.head()
```

Kode di atas menghasilkan *output* berikut:
```python
movie_df top 5 rows example
```

| movieId | title                             | genres                                          |
|---------|-----------------------------------|-------------------------------------------------|
| 1       | Toy Story (1995)                  | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 2       | Jumanji (1995)                    | Adventure\|Children\|Fantasy                    |
| 3       | Grumpier Old Men (1995)           | Comedy\|Romance                                 |
| 4       | Waiting to Exhale (1995)          | Comedy\|Drama\|Romance                          |
| 5       | Father of the Bride Part II (1995)| Comedy                                          |

Berdasarkan *output* kode di atas didapatkan informasi berikut:
- fitur `movieId` dan `title` sudah sesuai dan dapat lanjut ke tahap berikutnya.
- fitur `genres` berisi lebih dari satu genre untuk tiap film dan perlu diproses dengan mengambil genre utama dari tiap film.

Melakukan split terhadap kolom *genres* pada `movie_df`.

```python
movie_df['genres'] = movie_df['genres'].str.split('|').str[0]
```

Kode di atas bertujuan untuk mengambil genre utama dari tiap film dan menjadikan kolom genre hanya berisi satu genre dengan melakukan split terhadap *value* dari tiap genre.

Melihat contoh data dari `movie_df` dengan mengambil 5 baris pertama.

```python
print('movie_df top 5 rows')
movie_df.head()
```

Kode di atas menghasilkan *output* berikut:
```python
movie_df top 5 rows
```

| movieId | title                             |   genres   |
|---------|-----------------------------------|------------|
| 0       | Toy Story (1995)                  | Adventure  |
| 1       | Jumanji (1995)                    | Adventure  |
| 2       | Grumpier Old Men (1995)           | Comedy     |
| 3       | Waiting to Exhale (1995)          | Comedy     |
| 4       | Father of the Bride Part II (1995)| Comedy     |

Melihat jumlah data `movieId` yang unik yang ada di `movie_df`.

```python
print('Num of unique movieId: ', len(movie_df['movieId'].unique()))
```

Kode di atas menghasilkan *output* berikut:

```python
Num of unique movieId:  27278
```

Kode di atas bertujuan untuk mengecek apakah kolom `movieId` yang ada pada `movie_df` unik semua dan tidak ada yang sama. Bisa dilihat terdapat 27278 data unik yang dimana sama dengan jumlah data, yang dimana berarti seluruh data `movieId` yang ada pada `movie_df` merupakan data unik.

Melihat jumlah data genre dari `movie_df` dan menampilkannya.

```python
print('Num of unique genres:', len(movie_df['genres'].unique()))
print('List of unique genres:', movie_df['genres'].unique())
```

Kode di atas menghasilkan *output* berikut:
```python
Num of unique genres: 20
List of unique genres: ['Adventure' 'Comedy' 'Action' 'Drama' 'Crime' 'Children' 'Mystery'
 'Documentary' 'Animation' 'Thriller' 'Horror' 'Fantasy' 'Western'
 'Film-Noir' 'Romance' 'War' 'Sci-Fi' 'Musical' 'IMAX'
 '(no genres listed)']
```

Dari *output* kode di atas, dapat dilihat terdapat 20 genre yang unik dan terdiri atas:
- Adventure
- Comedy
- Action
- Drama
- Crime
- Children
- Mystery
- Documentary
- Animation
- Thriller
- Horror
- Fantasy
- Western
- Film-Noir
- Romance
- War
- Sci-Fi
- Musical
- IMAX
- (no genres listed)

Melihat jumlah data dari setiap genre yang ada di kolom `genres` yang ada di `movie_df`.

```python
movie_df['genres'].value_counts()
```

Kode di atas menghasilkan *output* berikut:

| Genre                | Count  |
|----------------------|--------|
| Drama                | 7,875  |
| Comedy               | 6,793  |
| Action               | 3,520  |
| Documentary          | 2,247  |
| Crime                | 1,617  |
| Horror               | 1,380  |
| Adventure            | 1,357  |
| Animation            | 571    |
| Children             | 414    |
| Thriller             | 279    |
| (no genres listed)   | 246    |
| Western              | 215    |
| Sci-Fi               | 168    |
| Mystery              | 146    |
| Romance              | 145    |
| Fantasy              | 137    |
| Musical              | 94     |
| Film-Noir            | 38     |
| War                  | 35     |
| IMAX                 | 1      |

Dari *output* kode di atas, dapat dilihat bahwa terdapat *outlier* pada `movie_df` yang dimana nantinya akan dihapus.

Mengambil panjang baris dan kolom serta nama-nama kolom yang ada di `rating_df`.

```python
print('rating_df row and column length:', rating_df.shape)
print('rating_df columns: ', rating_df.keys())
```

Kode di atas menghasilkan *output* berikut:

rating_df row and column length: (20000263, 4)
rating_df columns:  Index(['userId', 'movieId', 'rating', 'timestamp'], dtype='object')

Berdasarkan *output* kode di atas didapatkan informasi berikut:

*   Jumlah baris dan kolom adalah 20000263 baris dan 4 kolom.
*   Terdapat 9 kolom fitur yaitu `userId`, `movieId`, `rating`, `timestamp`.

Mengambil informasi dari kolom-kolom yang ada di `rating_df`.

```python
print('rating_df columns information')
rating_df.info()
```

Kode di atas menghasilkan *output* berikut:

```python
rating_df columns information
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20000263 entries, 0 to 20000262
Data columns (total 4 columns):
 #   Column     Dtype  
---  ------     -----  
 0   userId     int64  
 1   movieId    int64  
 2   rating     float64
 3   timestamp  object 
dtypes: float64(1), int64(2), object(1)
memory usage: 610.4+ MB
```

Berdasarkan *output* kode di atas didapatkan informasi berikut:

*   Terdapat 1 kolom dengan tipe *object*, yaitu: `timestamp`. Kolom ini merupakan categorical features (fitur non-numerik).
*   Terdapat 2 kolom numerik dengan tipe data *int64*, yaitu: `userId` dan `movieId`.
*   Terdapat 1 kolom numerik dengan tipe data *float64*, yaitu `rating`.

Mengambil informasi statistik dari `rating_df`.

```python
print('rating_df statistic information')
rating_df.describe()
```

Kode di atas menghasilkan *output* berikut:

| Statistic | userId           | movieId          | rating           |
|-----------|------------------|------------------|------------------|
| Count     | 20,000,260       | 20,000,260       | 20,000,260       |
| Mean      | 69,045.87        | 9,041.57         | 3.53             |
| Std       | 40,038.63        | 19,789.48        | 1.05             |
| Min       | 1.00             | 1.00             | 0.50             |
| 25%       | 34,395.00        | 902.00           | 3.00             |
| 50%       | 69,141.00        | 2,167.00         | 3.50             |
| 75%       | 103,637.00       | 4,770.00         | 4.00             |
| Max       | 138,493.00       | 131,262.00       | 5.00             |

Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:
*   **Count**  adalah jumlah sampel pada data.
*   **Mean** adalah nilai rata-rata.
*   **Std** adalah standar deviasi.
*   **Min** yaitu nilai minimum setiap kolom.
*   **25%** adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
*   **50%** adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
*   **75%** adalah kuartil ketiga.
*   **Max** adalah nilai maksimum.

Melihat contoh data dari `rating_df` dengan mengambil 5 baris pertama.

```python
print('rating_df top 5 rows example')
rating_df.head()
```

Kode di atas menghasilkan *output* berikut:

| userId | movieId | rating | timestamp           |
|--------|---------|--------|---------------------|
| 1      | 2       | 3.5    | 2005-04-02 23:53:47 |
| 1      | 29      | 3.5    | 2005-04-02 23:31:16 |
| 1      | 32      | 3.5    | 2005-04-02 23:33:39 |
| 1      | 47      | 3.5    | 2005-04-02 23:32:07 |
| 1      | 50      | 3.5    | 2005-04-02 23:29:40 |

Melihat jumlah data user yang unik yang ada di `rating_df`.

```python
print('Num of unique users:', len(rating_df['userId'].unique()))
```

Kode di atas menghasilkan *output* berikut:

```python
Num of unique users: 138493
```

Berdasarkan *output* kode di atas, terdapat sebanyak 138493 data *user* yang unik pada `rating_df`.

Melihat jumlah data `movieId` yang unik yang ada di `rating_df`.

```python
print('Num of unique movie:', len(rating_df['movieId'].unique()))
```

Kode di atas menghasilkan *output* berikut:

```python
Num of unique users: 138493
```

Berdasarkan *output* kode di atas, terdapat sebanyak 138493 data *user* yang unik pada `rating_df`.

Melihat jumlah data `movieId` yang unik yang ada di `rating_df`.

```python
print('Num of unique movie:', len(rating_df['movieId'].unique()))
```

Kode di atas menghasilkan *output* berikut:

```python
Num of unique movie: 26744
```

Berdasarkan *output* kode di atas, terdapat sebanyak 138493 data film yang unik pada `rating_df`.

Melihat jumlah data yang memiliki value `null` pada `movie_df`.

```python
missing_values = movie_df.isnull().sum()
missing_values
```

Kode di atas menghasilkan *output* berikut:

| Column  | Missing Values |
|---------|----------------|
| movieId | 0              |
| title   | 0              |
| genres  | 0              |

Berdasarkan *output* kode di atas, tidak terdapat satupun data *null* pada `movie_df`.

Melihat jumlah data yang memiliki value `null` pada `rating_df`.

```python
missing_values = rating_df.isnull().sum()
missing_values
```

Berdasarkan *output* kode di atas, tidak terdapat satupun data *null* pada `rating_df`.

```python
print(movie_df.duplicated().any())
print(movie_df[movie_df.duplicated()])
```

Kode di atas menghasilkan *output* berikut:

```python
False
Empty DataFrame
Columns: [movieId, title, genres]
Index: []
```

Berdasarkan *output* kode di atas, tidak terdapat satupun data duplikat pada `movie_df`.

Melihat apakah terdapat data yang duplikat pada `rating_df`.

```python
print(rating_df.duplicated().any())
print(rating_df[rating_df.duplicated()])
```

Kode di atas menghasilkan *output* berikut:

```python
False
Empty DataFrame
Columns: [userId, movieId, rating, timestamp]
Index: []
```

Melihat apakah terdapat data yang memiliki *value* `NaN` pada `movie_df`.

```python
print(movie_df.isna().any().any())
print(movie_df[movie_df.isna().any(axis=1)])
```

Kode di atas menghasilkan *output* berikut:

```python
False
Empty DataFrame
Columns: [movieId, title, genres]
Index: []
```

Berdasarkan *output* kode di atas, tidak terdapat satupun data NaN pada `movie_df`.

Melihat apakah terdapat data yang memiliki *value* `NaN` pada `rating_df`.

```python
print(rating_df.isna().any().any())
print(rating_df[rating_df.isna().any(axis=1)])
```

Kode di atas menghasilkan *output* berikut:
```python
False
Empty DataFrame
Columns: [userId, movieId, rating, timestamp]
Index: []
```

Berdasarkan *output* kode di atas, tidak terdapat satupun data NaN pada `rating_df`.

### *Data Visualization*
#### *Univariate Analysis*
> *Univariate analysis* adalah jenis analisis yang berfokus pada satu variabel (*feature*) dalam satu waktu. Tujuan utamanya adalah untuk memahami distribusi, karakteristik, dan pola dari variabel tersebut.

Membuat fungsi untuk menghitung jumlah dan persentase data dari masing-masing data.

```python
def CountAndPlot(df, feature):
  count = df[feature].value_counts()
  percent = 100*df[feature].value_counts(normalize=True)
  samples = pd.DataFrame({'Sample Count':count, 'Percentage':percent.round(1)})
  print(samples)
  count.plot(kind='bar', title=feature)
```

Membuat *graph* untuk kolom genres pada `movie_df`.

```python
CountAndPlot(movie_df, 'genres')
```

Kode di atas menghasilkan *output* berikut:

| Genre                 | Sample Count | Percentage (%) |
|-----------------------|--------------|----------------|
| Drama                | 7,875        | 28.9           |
| Comedy               | 6,793        | 24.9           |
| Action               | 3,520        | 12.9           |
| Documentary          | 2,247        | 8.2            |
| Crime                | 1,617        | 5.9            |
| Horror               | 1,380        | 5.1            |
| Adventure            | 1,357        | 5.0            |
| Animation            | 571          | 2.1            |
| Children             | 414          | 1.5            |
| Thriller             | 279          | 1.0            |
| (no genres listed)   | 246          | 0.9            |
| Western              | 215          | 0.8            |
| Sci-Fi               | 168          | 0.6            |
| Mystery              | 146          | 0.5            |
| Romance              | 145          | 0.5            |
| Fantasy              | 137          | 0.5            |
| Musical              | 94           | 0.3            |
| Film-Noir            | 38           | 0.1            |
| War                  | 35           | 0.1            |
| IMAX                 | 1            | 0.0            |

![Graph - Genres in movie_df](https://raw.githubusercontent.com/fikridean/Movie-Recommendation-System/refs/heads/main/graph-genres_in_movie_df.png)

Berdasarkan grafik di atas, dapat dilihat bahwa jumlah data genre drama mendominasi `movie_df` yang dimana disusul oleh genre `comedy` dan `action`.

```python
CountAndPlot(rating_df, 'rating')
```

## Rating Distribution with Sample Count and Percentage

| Rating | Sample Count | Percentage (%) |
|--------|--------------|----------------|
| 4.0    | 5,561,926    | 27.8           |
| 3.0    | 4,291,193    | 21.5           |
| 5.0    | 2,898,660    | 14.5           |
| 3.5    | 2,200,156    | 11.0           |
| 4.5    | 1,534,824    | 7.7            |
| 2.0    | 1,430,997    | 7.2            |
| 2.5    | 883,398      | 4.4            |
| 1.0    | 680,732      | 3.4            |
| 1.5    | 279,252      | 1.4            |
| 0.5    | 239,125      | 1.2            |

![Graph - Genres in movie_df](https://raw.githubusercontent.com/fikridean/Movie-Recommendation-System/refs/heads/main/graph-genres_in_movie_df.png)

Berdasarkan grafik di atas, dapat dilihat bahwa mayoritas *user* memberikan rating ke film adalah sebesar `4.0`.

## Data Preparation
### *Data Cleaning*

> **Penjelasan**: *Data Cleaning* adalah proses untuk memperbaiki atau menghapus data yang salah, rusak, tidak sesuai format, duplikat, atau tidak lengkap dalam sebuah *dataset*. Ketika berbagai sumber data digabungkan, risiko terjadinya duplikasi atau pelabelan yang salah meningkat secara signifikan.

> **Alasan**: Proses *Data Cleaning* penting dilakukan untuk memeriksa, memperbaiki, dan memastikan bahwa data yang digunakan siap diproses serta bebas dari potensi kesalahan.

#### *Removal Duplicates and NaN Data*

> **Penjelasan**: Menghapus data duplikat bertujuan memastikan dataset menjadi unik dan bebas dari baris yang redundan. Duplikasi dapat terjadi karena kesalahan dalam pengumpulan data, penggabungan dataset, atau duplikasi yang tidak disengaja.

> **Alasan**: Proses ini diperlukan untuk membersihkan data yang dapat memicu kesalahan, seperti nilai *NaN*, dan menghilangkan data yang berpotensi menurunkan performa, seperti data duplikat.

Membersihkan `movie_df` dan `rating_df` dari data yang duplikat dan data yang memiliki *value* `NaN`. Kemudian memasukkan hasil dari kedua proses tersebut ke `clean_movie_df` dan `clean_movie_df`.

```python
clean_movie_df = movie_df.drop_duplicates().dropna()
clean_rating_df = rating_df.drop_duplicates().dropna()
```

```python
clean_movie_df.head()
```

Kode di atas menghasilkan *output* berikut:

| movieId | title                              | genres     |
|---------|------------------------------------|------------|
| 1       | Toy Story (1995)                  | Adventure  |
| 2       | Jumanji (1995)                    | Adventure  |
| 3       | Grumpier Old Men (1995)           | Comedy     |
| 4       | Waiting to Exhale (1995)          | Comedy     |
| 5       | Father of the Bride Part II (1995)| Comedy     |

```python
clean_rating_df.head()
```

Kode di atas menghasilkan *output* berikut:

| userId | movieId | rating | timestamp           |
|--------|---------|--------|---------------------|
| 1      | 2       | 3.5    | 2005-04-02 23:53:47 |
| 1      | 29      | 3.5    | 2005-04-02 23:31:16 |
| 1      | 32      | 3.5    | 2005-04-02 23:33:39 |
| 1      | 47      | 3.5    | 2005-04-02 23:32:07 |
| 1      | 50      | 3.5    | 2005-04-02 23:29:40 |

#### *Removal Irrelevant Columns*

> **Penjelasan**: Menghapus data yang tidak relevan dan tidak diperlukan pada proyek.

> **Alasan**: Proses ini diperlukan untuk membersihkan data dari data yang tidak relevan dan tidak diperlukan. Kemudian selanjutnya memastikan bahwa `movieId` yang ada di `rating_df` ada di `movie_df` juga agar sesuai ketika data diproses pada tahap-tahap selanjutnya.

Jika dilihat dari `plot` pada tahap *data understanding*, pada `movie_df` terdapat satu genre yang dimana perlu untuk dihapus karena tidak diperlukan untuk pengembangan model, yaitu genre `(no genres listed)`. Selain itu, pada `rating_df` terdapat kolom `timestamp` yang dimana tidak diperlukan pada pengembangan model.

Menghapus genre film yang tidak relevan yaitu '(no genres listed)' dari `clean_movie_df`.

```python
clean_movie_df = clean_movie_df.loc[clean_movie_df['genres'] != '(no genres listed)']
clean_movie_df['genres'].value_counts()
```

Kode di atas menghasilkan *output* berikut:

| Genre       | Count |
|-------------|-------|
| Drama       | 7,875 |
| Comedy      | 6,793 |
| Action      | 3,520 |
| Documentary | 2,247 |
| Crime       | 1,617 |
| Horror      | 1,380 |
| Adventure   | 1,357 |
| Animation   | 571   |
| Children    | 414   |
| Thriller    | 279   |
| Western     | 215   |
| Sci-Fi      | 168   |
| Mystery     | 146   |
| Romance     | 145   |
| Fantasy     | 137   |
| Musical     | 94    |
| Film-Noir   | 38    |
| War         | 35    |
| IMAX        | 1     |

Menghapus kolom `timestamp` pada `clean_rating_df` yang dimana tidak diperlukan untuk pengembangan model.

```python
clean_rating_df = clean_rating_df.drop(columns=['timestamp'])
clean_rating_df.head()
```

| userId | movieId | rating |
|--------|---------|--------|
| 1      | 2       | 3.5    |
| 1      | 29      | 3.5    |
| 1      | 32      | 3.5    |
| 1      | 47      | 3.5    |
| 1      | 50      | 3.5    |

Kemudian selanjutnya memastikan bahwa `movieId` yang ada di `rating_df` terdaftar pada `movie_df` juga agar sesuai ketika data diproses pada tahap-tahap selanjutnya.

```python
movieIds = clean_rating_df['movieId'].unique()
print('Total: ', len(movieIds))
movieIds
```

Kode di atas menghasilkan *output* berikut:

```python
Total:  26744
array([     2,     29,     32, ..., 121021, 110167, 110510])
```

```python
len(clean_rating_df)
```

Kode di atas menghasilkan *output* berikut:

```python
20000263
```

```python
clean_rating_df = clean_rating_df[clean_rating_df['movieId'].isin(movieIds)]
len(clean_rating_df)
```

Kode di atas menghasilkan *output* berikut:

```python
20000263
```

#### *Handle Imbalance Data*

> **Penjelasan**: Memeriksa dan menghapus data untuk mengatasi masalah ketidakseimbangan

> **Alasan**: Proses ini diperlukan untuk membersihkan data dari data yang terlalu sedikit jumlahnya dan tidak seimbang terhadap jumlah data lainnya

Menghapus data genre *IMAX* karena terlalu sedikit data yang ada pada `clean_movie_df`.

```python
clean_movie_df = clean_movie_df[clean_movie_df['genres'] != 'IMAX']
print('IMAX Genres Count : ', len(clean_movie_df[clean_movie_df['genres'] == 'IMAX']))
print()
print(clean_movie_df['genres'].value_counts())
```

Kode di atas menghasilkan *output* berikut:

| Genre       | Count |
|-------------|-------|
| Drama       | 7,875 |
| Comedy      | 6,793 |
| Action      | 3,520 |
| Documentary | 2,247 |
| Crime       | 1,617 |
| Horror      | 1,380 |
| Adventure   | 1,357 |
| Animation   | 571   |
| Children    | 414   |
| Thriller    | 279   |
| Western     | 215   |
| Sci-Fi      | 168   |
| Mystery     | 146   |
| Romance     | 145   |
| Fantasy     | 137   |
| Musical     | 94    |
| Film-Noir   | 38    |
| War         | 35    |
| IMAX        | 0     |

Melakukan *grouping* untuk setiap `movieId` terhadap seluruh data *rating* yang ada.

```python
grouped_ratings = clean_rating_df.groupby('movieId')
grouped_ratings.groups
```

Kode di atas menghasilkan *output* berikut:
```python
{1: [236, 517, 817, 922, 960, 1464, 1500, 1562, 1854, 2061, 2302, 2435, 2537, 3534, 3903, 4305, 4794, 5194, 5400, 6343, 7000, 7477, ...], ...}
```

Melakukan *undersampling* dengan masing-masing `movie_id` akan dikaitkan dengan maksimal tiga `userId`.

```python
target_ratings = 3

clean_rating_df = pd.concat(
    [
        group.sample(n=target_ratings, random_state=42) if len(group) > target_ratings else
        resample(group, replace=True, n_samples=target_ratings, random_state=42)
        for _, group in grouped_ratings
    ]
)

clean_rating_df
```

Kode di atas menghasilkan *output* berikut:

### *Text Processing*

> **Penjelasan**: Proses mengidentifikasi dan memperbaiki atau menghapus data yang salah, tidak konsisten, atau tidak relevan dalam dataset.
> **Alasan**: Proses ini diperlukan agar semua data konsisten dan tidak ada data yang dalam format berbeda

Memperbaiki dan menyelaraskan format penulisan untuk genre-genre film.

```python
clean_movie_df['genres'] = clean_movie_df['genres'].replace({'Film-Noir': 'Filmnoir', 'Sci-Fi': 'Scifi'})
print('IMAX Genres Count : ', len(clean_movie_df[clean_movie_df['genres'] == 'Filmnoir']))
print('IMAX Genres Count : ', len(clean_movie_df[clean_movie_df['genres'] == 'Scifi']))
```

Kode di atas menghasilkan *output* berikut:

```python
IMAX Genres Count :  38
IMAX Genres Count :  168
```

```python
print('Example movie with genre Filmnoir')
print(clean_movie_df[clean_movie_df['genres'] == 'Filmnoir'].head())
```

Kode di atas menghasilkan *output* berikut:

```python
Example movie with genre Filmnoir
```

| movieId | title                           | genres    |
|---------|---------------------------------|-----------|
| 320     | Suture (1993)                   | Film-Noir |
| 746     | Force of Evil (1948)            | Film-Noir |
| 913     | Maltese Falcon, The (1941)      | Film-Noir |
| 930     | Notorious (1946)                | Film-Noir |
| 1153    | Raw Deal (1948)                 | Film-Noir |

```python
print('Example movie with genre Scifi')
print(clean_movie_df[clean_movie_df['genres'] == 'Scifi'].head())
```

Kode di atas menghasilkan *output* berikut:

```python
Example movie with genre Filmnoir
```

| movieId | title                                 | genres |
|---------|---------------------------------------|--------|
| 880     | Island of Dr. Moreau, The (1996)      | Sci-Fi |
| 1779    | Sphere (1998)                         | Sci-Fi |
| 2311    | 2010: The Year We Make Contact (1984) | Sci-Fi |
| 2526    | Meteor (1979)                         | Sci-Fi |
| 2578    | Sticky Fingers of Time, The (1997)    | Sci-Fi |

### *Data Transformation*
> **Penjelasan**: *Data transformation* adalah proses mengubah data mentah ke dalam format yang lebih cocok untuk analisis atau pelatihan model. Transformasi ini melibatkan berbagai teknik, seperti normalisasi, standardisasi, encoding, dan scaling, untuk memastikan data berada dalam skala atau bentuk yang sesuai dengan kebutuhan algoritma machine learning.

> **Alasan**: Proses ini diperlukan untuk meningkatkan kualitas data sehingga model dapat belajar secara lebih efektif seperti dalam pengembangan model *sentiment analysis*.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
```

```python
tfidf_matrix = tfidf.fit_transform(movie_df['genres'])
tfidf_matrix.shape
```

Kode di atas menghasilkan *output* berikut:
```python
(27278, 24)
```

```python
tfidf_matrix.todense()
```

Kode di atas menghasilkan *output* berikut:
```python
matrix([[0., 1., 0., ..., 0., 0., 0.],
        [0., 1., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        ...,
        [0., 1., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        [0., 1., 0., ..., 0., 0., 0.]])
```

### *Calculating Cosine Similarity*
> **Penjelasan**: *Cosine similarity* adalah metode yang digunakan untuk mengukur tingkat kesamaan antara dua vektor dalam ruang multidimensi berdasarkan sudut di antara mereka. Nilai cosine similarity berkisar antara -1 dan 1, di mana nilai 1 menunjukkan kesamaan sempurna, 0 menunjukkan tidak ada kesamaan, dan -1 menunjukkan arah yang sepenuhnya berlawanan.

> **Alasan**: Proses ini harus dilakukan karena kemampuannya untuk mengukur kesamaan antara dokumen atau data yang akan digunakan pada model.

```python
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim
```

Kode di atas menghasilkan *output* berikut:
```python
array([[1., 1., 0., ..., 1., 0., 1.],
       [1., 1., 0., ..., 1., 0., 1.],
       [0., 0., 1., ..., 0., 0., 0.],
       ...,
       [1., 1., 0., ..., 1., 0., 1.],
       [0., 0., 0., ..., 0., 1., 0.],
       [1., 1., 0., ..., 1., 0., 1.]])
```

### *Data Encoding*
> **Penjelasan**: *Data encoding* adalah proses mengubah data menjadi format numerik.

> **Alasan**: Sebelumnya *value* dari userId dan movieId tidak beurutan (0, 1, 2, dst) sehingga proses ini proses ini harus dilakukan untuk mengubah data menjadi numerik yang berurutan (*index*) agar model dapat dengan mudah diproses.

```python
user_ids = clean_rating_df['userId'].unique().tolist()
print('list userId: ', user_ids)

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userID : ', user_to_user_encoded)

user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userID: ', user_encoded_to_user)
```

Kode di atas menghasilkan *output* berikut:
```python
list userId:  [33218, 11586, ...]
encoded userID: {33218: 0, 11586: 1, 49444: 2, ...}
encoded angka ke userID: {0: 33218, 1: 11586, 2: 49444, ...}
```

```python
movie_ids = clean_rating_df['movieId'].unique().tolist()
print('list movieId: ', movie_ids)

movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
print('encoded movieId : ', movie_to_movie_encoded)

movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}
print('encoded angka ke movieId: ', movie_encoded_to_movie)
```

Kode di atas menghasilkan *output* berikut:
```python
list movieId:  [1, 2, ...]
encoded movieId: {1: 0, 2: 1, ...}
encode angka: {0: 1, 1: 2, ...}
```

```python
clean_rating_df['user'] = clean_rating_df['userId'].map(user_to_user_encoded)
clean_rating_df['movie'] = clean_rating_df['movieId'].map(movie_to_movie_encoded)
clean_rating_df
```

Kode di atas menghasilkan *output* berikut:
| Index     | userId    | movieId   | rating | user  | movie  |
|-----------|-----------|-----------|--------|-------|--------|
| 4834169   | 33218     | 1         | 5.0    | 0     | 0      |
| 1716733   | 11586     | 1         | 3.0    | 1     | 0      |
| 7170935   | 49444     | 1         | 4.0    | 2     | 0      |
| 2461714   | 16670     | 2         | 2.0    | 3     | 1      |
| 12736079  | 88015     | 2         | 3.5    | 4     | 1      |
| ...       | ...       | ...       | ...    | ...   | ...    |
| 9459645   | 65409     | 131260    | 3.0    | 14538 | 26742  |
| 9459645   | 65409     | 131260    | 3.0    | 14538 | 26742  |
| 19227262  | 133047    | 131262    | 4.0    | 19585 | 26743  |
| 19227262  | 133047    | 131262    | 4.0    | 19585 | 26743  |
| 19227262  | 133047    | 131262    | 4.0    | 19585 | 26743  |

```python
num_users = len(user_to_user_encoded)
print(num_users)

num_movie = len(movie_encoded_to_movie)
print(num_movie)

clean_rating_df['rating'] = clean_rating_df['rating'].values.astype(np.float32)
min_rating = min(clean_rating_df['rating'])
max_rating = max(clean_rating_df['rating'])

print('Number of User: {}, Number of Movie: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_movie, min_rating, max_rating
))
```

Kode di atas menghasilkan *output* berikut:
```python
20404
26744
Number of User: 20404, Number of Movie: 26744, Min Rating: 0.5, Max Rating: 5.0
```

### Train Test split
> **Penjelasan**: *Train-test split* adalah metode sederhana untuk mengukur kinerja algoritma machine learning dalam prediksi dengan membandingkan hasil model yang dibuat dengan data yang belum pernah dilihat sebelumnya.

> **Alasan**: Proses ini dilakukan dengan tujuan untuk melakukan evaluasi terhadap model yang akan dibuat dengan mengukur seberapa baik model dapat memprediksi data yang belum pernah dilihat (data yang tidak dimasukkan ke data latih melainkan ke data *test*).

```python
clean_rating_df = clean_rating_df.sample(frac=1, random_state=42)
clean_rating_df

x = clean_rating_df[['user', 'movie']].values
y = clean_rating_df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * clean_rating_df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)
```

Kode di atas menghasilkan *output* berikut:
```python
[[ 1154 21338]
 [15268  9984]
 [  991  4567]
 ...
 [ 4449 25606]
 [  831   286]
 [ 8479  5265]] [0.66666667 1.         0.55555556 ... 0.44444444 0.55555556 0.33333333]
```

## Modeling and Result
Pada proyek ini, terdapat dua model sistem rekomendasi yang dibuat yaitu model sistem rekomendasi dengan teknik *Content Based Filtering* dan dengan teknik *Collaborative Filtering*.

1. *Content Based Filtering*

    *Content-Based Filtering* merekomendasikan sesuatu (*item*) kepada pengguna berdasarkan apa yang sudah mereka sukai sebelumnya. Teknik ini fokus pada karakteristik atau fitur dari *item* tersebut.

    Kelebihan:
      - Personal: Rekomendasinya benar-benar sesuai dengan minat pengguna.
      - Mudah dikembangkan: Jika terdapat informasi baru berkaitan dengan *item*, sistem bisa langsung diperbarui.
      - Tidak perlu data pengguna lain: Hanya memerlukan data pengguna itu sendiri.

    Kekurangan:
      - Pilihan terbatas: Hanya bisa merekomendasikan item yang mirip dengan yang sudah disukai, jadi kurang variasi.
      - Masalah untuk item baru: Jika *item* belum punya deskripsi lengkap, sulit untuk direkomendasikan.


2. *Collaborative Filtering*
    *Collaborative Filtering* merekomendasikan sesuatu (*item*) berdasarkan kesukaan orang lain yang memiliki preferensi mirip dengan pengguna.

    Kelebihan:
      - Variasi lebih besar: Teknik ini daoat merekomendasikan hal-hal yang belum pernah dicoba pengguna, karena datanya berasal dari pengguna lain.
      - Tidak perlu tahu detail *item*: Teknik ini hanya melihat pola kesukaan dari pengguna-pengguna sebelumnya.

    Kekurangan:
      - Butuh banyak data interaksi: Jika data pengguna sedikit, teknik ini sulit untuk digunakan.
      - Masalah pengguna baru: Jika pengguna baru belum pernah memberikan *rating*, sistem tidak tahu harus merekomendasikan apa.
      - Komputasi berat: Jika data yang digunakan cukup besar, membandingkan banyak pengguna atau *item* dapat membutuhkan waktu yang cukup lama dan sumber daya yang cukup besar.

### *Modelling with Content Based Filtering Technique*
Membuat fungsi yang akan digunakan untuk menghasilkan *Top-N recommendation film* sebagai *output* berdasarkan nama film sebagai *input*.

```python
def movie_recommendations(movie_name, similarity_data=cosine_sim_df, items=movie_df[['title', 'genres']], k=5):
    index = similarity_data.loc[:,movie_name].to_numpy().argpartition(
        range(-1, -k, -1))

    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    closest = closest.drop(movie_name, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)
```

Memeriksa apakah judul yang akan dicoba sebagai *input* terdapat di `movie_df`.

```python
movie_df[movie_df.title.eq('Hell in the Pacific (1968)')]
```

Kode di atas menghasilkan *output* berikut:

| movieId | title                        | genres |
|---------|------------------------------|--------|
| 3143    | Hell in the Pacific (1968)  | Drama  |

Menjalankan model yang dimana nama film *Hell in the Pacific (1968)* dijadikan sebagai *input*.

```python
movie_recommendations('Hell in the Pacific (1968)')
```

| Index | title                                       | genres |
|-------|---------------------------------------------|--------|
| 0     | Traviata, La (1982)                         | Drama  |
| 1     | Screaming Man, A (Un homme qui crie) (2010) | Drama  |
| 2     | Book of Stars, The (1999)                   | Drama  |
| 3     | May 18 (Hwaryeohan hyuga) (2007)            | Drama  |
| 4     | Show Me Love (Fucking Åmål) (1998)          | Drama  |

Berdasarkan *output* dari kode di atas, terdapat lima film yang dihasilkan dari model sistem rekomendasi dengan teknik *content based filtering*.

### *Modelling with Collaborative Filtering Technique*
Membuat fungsi yang akan dijadikan fondasi utama untuk model dengan teknik *collaborative filtering technique*.

```python
class RecommenderNet(tf.keras.Model):

  def __init__(self, num_users, num_movie, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_movie = num_movie
    self.embedding_size = embedding_size
    self.user_embedding = tf.keras.layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = tf.keras.regularizers.l2(1e-6)
    )
    self.user_bias = tf.keras.layers.Embedding(num_users, 1)
    self.resto_embedding = tf.keras.layers.Embedding(
        num_movie,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = tf.keras.regularizers.l2(1e-6)
    )
    self.resto_bias = tf.keras.layers.Embedding(num_movie, 1)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0])
    user_bias = self.user_bias(inputs[:, 0])
    resto_vector = self.resto_embedding(inputs[:, 1])
    resto_bias = self.resto_bias(inputs[:, 1])

    dot_user_resto = tf.tensordot(user_vector, resto_vector, 2)

    x = dot_user_resto + user_bias + resto_bias

    return tf.nn.sigmoid(x)
```

Inisialisasi model menggunakan fungsi yang telah dibuat sebelumnya.

```python
model = RecommenderNet(num_users, num_movie, 50)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```

Melakukan proses *training* pada model yang telah diinisialisasikan sebelumnya.

```python
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 10,
    validation_data = (x_val, y_val)
)
```

Kode di atas menghasilkan *output* berikut:
```python
Epoch 1/10
8024/8024 ━━━━━━━━━━━━━━━━━━━━ 25s 3ms/step - loss: 0.6870 - root_mean_squared_error: 0.2473 - val_loss: 0.6736 - val_root_mean_squared_error: 0.2336
...
Epoch 10/10
8024/8024 ━━━━━━━━━━━━━━━━━━━━ 38s 2ms/step - loss: 0.6044 - root_mean_squared_error: 0.1503 - val_loss: 0.6505 - val_root_mean_squared_error: 0.2095
```

Setelah model berhasil dilatih, selanjutnya adalah mendapatkan hasil dari prediksi model sistem rekomendasi.

```python
user_id = clean_rating_df.userId.sample(1).iloc[0]
movie_watched_by_user = clean_rating_df[clean_rating_df.userId == user_id]
movie_not_watched = clean_movie_df[~clean_movie_df['movieId'].isin(movie_watched_by_user.movieId.values)]['movieId']

movie_not_watched = list(
    set(movie_not_watched)
    .intersection(set(movie_to_movie_encoded.keys()))
)

movie_not_watched = [[movie_to_movie_encoded.get(x)] for x in movie_not_watched]
user_encoder = user_to_user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_watched), movie_not_watched)
)

ratings = model.predict(user_movie_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded_to_movie.get(movie_not_watched[x][0]) for x in top_ratings_indices
]

print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('movie with high ratings from user')
print('----' * 8)

top_movie_user = (
    movie_watched_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .movieId.values
)

movie_df_rows = movie_df[movie_df['movieId'].isin(top_movie_user)]
for row in movie_df_rows.itertuples():
    print(row.title)

print('----' * 8)
print('Top 10 movie recommendation')
print('----' * 8)

recommended_movie = movie_df[movie_df['movieId'].isin(recommended_movie_ids)]
for row in recommended_movie.itertuples():
    print(row.title)
```

Kode di atas menghasilkan *output* berikut:
```python
829/829 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step
Showing recommendations for users: 53498
===========================
movie with high ratings from user
--------------------------------
Halloween: The Curse of Michael Myers (Halloween 6: The Curse of Michael Myers) (1995)
Autopsy (Macchie Solari) (1975)
Exorcist: The Beginning (2004)
Fourth Kind, The (2009)
--------------------------------
Top 10 movie recommendation
--------------------------------
My Life as a Dog (Mitt liv som hund) (1985)
Kundun (1997)
Cowboy Bebop: The Movie (Cowboy Bebop: Tengoku no Tobira) (2001)
Taste of Tea, The (Cha no aji) (2004)
Best of Ernie and Bert, The (1988)
Limuzins Janu nakts krasa (1981)
Boyhood (2014)
New Rulers of the World, The (2001)
Aashiqui 2 (2013)
In Name Only (1939)
```

## Evaluation
Setelah tahap *modelling* telah dilalui, selanjutnya adalah tahap *evaluation* yang dimana model yang telah dibuat akan dievaluasi kinerjanya. Pada model yang menggunakan teknik *content based filtering*, akan digunakan metrik evaluasi yaitu *Precision*. Kemudian untuk model yang menggunakan teknik *collaborative filtering*, akan digunakan metrik *Root Mean Squared Error* (RMSE) sebagai metrik evaluasinya.

### Metrik *Model with Content Based Filtering* (Metrik *Precision*)

- Penjelasan *Precision*
  - Mengukur seberapa akurat model dalam memprediksi kelas positif dengan menunjukkan seberapa banyak prediksi positif yang benar dari semua prediksi positif.

- Rumus *Precision*
  - **Precision = TP / (TP + FP)**
  - TP (*True Positives*): Kasus di mana model memprediksi "positif" dan prediksinya benar.
  - FP (*False Positives*): Kasus di mana model memprediksi "positif" tetapi prediksinya salah.

- Cara Kerja
  - Precision mengukur proporsi prediksi positif yang benar dari seluruh prediksi positif yang dihasilkan. Cara kerjanya adalah dengan membandingkan jumlah *True Positives* (TP) dengan total prediksi positif, yaitu *True Positives* (TP) + *False Positives* (FP). Precision menunjukkan seberapa relevan hasil positif yang diberikan oleh model, sehingga cocok digunakan saat fokus utama adalah memastikan kualitas hasil yang akurat. *italicized text*

```python
def evaluate_content_based_filtering(movie_name, similarity_data=cosine_sim_df, items=movie_df[['title', 'genres']], k=5):
    recommended_movies = movie_recommendations(movie_name, similarity_data, items, k)['title'].values
    input_movie_genres = movie_df[movie_df['title'] == movie_name]['genres'].values[0].split('|')

    relevant = [
        1 if any(genre in input_movie_genres for genre in movie_df[movie_df['title'] == movie]['genres'].values[0].split('|'))
        else 0
        for movie in recommended_movies
    ]

    precision = np.mean(relevant)

    relevant_movies = [movie for movie, is_relevant in zip(recommended_movies, relevant) if is_relevant == 1]
    irrelevant_movies = [movie for movie, is_relevant in zip(recommended_movies, relevant) if is_relevant == 0]

    relevant_count = len(relevant_movies)
    irrelevant_count = len(irrelevant_movies)

    evaluation_summary = {
        'Movie': movie_name,
        'Total Recommendations': k,
        'Relevant Recommendations': relevant_count,
        'Irrelevant Recommendations': irrelevant_count,
        'Precision': precision,
        'Relevant Movies': relevant_movies,
        'Irrelevant Movies': irrelevant_movies
    }

    return evaluation_summary

evaluation = evaluate_content_based_filtering('Hell in the Pacific (1968)')
print(f"Evaluation Summary for 'Hell in the Pacific (1968)':")
print(f"Total Recommendations: {evaluation['Total Recommendations']}")
print(f"Relevant Recommendations: {evaluation['Relevant Recommendations']}")
print(f"Irrelevant Recommendations: {evaluation['Irrelevant Recommendations']}")
print(f"Precision: {evaluation['Precision']:.4f}")
print(f"Relevant Movies: {', '.join(evaluation['Relevant Movies'])}")
print(f"Irrelevant Movies: {', '.join(evaluation['Irrelevant Movies'])}")
```

Kode di atas menghasilkan *output* berikut:
```python
Evaluation Summary for 'Hell in the Pacific (1968)':
Total Recommendations: 5
Relevant Recommendations: 5
Irrelevant Recommendations: 0
Precision: 1.0000
Relevant Movies: Traviata, La (1982), Screaming Man, A (Un homme qui crie) (2010), Book of Stars, The (1999), May 18 (Hwaryeohan hyuga) (2007), Show Me Love (Fucking Åmål) (1998)
Irrelevant Movies:
``` 

Berdasarkan *output* kode evaluasi diatas, dapat dilihat bahwa setelah model dijalankan dengan memasukkan 'Hell in the Pacific (1968)' sebagai *input*, didapatkan lima rekomendasi film sebagai *output* dari model. Berdasarkan *output* tersebut, kelima film merupakan film yang relevan dari film yang dimasukkan sebagai *input* ke model yang dimana hal ini menunjukkan bahwa model memiliki precision yang sangat tinggi yaitu `1.0` atau `100%`.

### Metrik *Model with Content Based Filtering* (Metrik *Root Mean Squared Error*)

- Penjelasan *Root Mean Squared Error*
  - RMSE adalah metrik yang digunakan untuk mengukur seberapa besar rata-rata kesalahan antara nilai yang diprediksi oleh model dengan nilai yang sebenarnya. Nilai RMSE dinyatakan dalam satuan yang sama dengan data asli, sehingga mudah untuk diinterpretasikan. Semakin kecil nilai RMSE, semakin baik model dalam membuat prediksi.

- Rumus *Precision*
  - ![Rumus RMSE](https://raw.githubusercontent.com/fikridean/Movie-Recommendation-System/refs/heads/main/rumus-RMSE.png)
- Cara Kerja
  - RMSE dihitung dengan membandingkan nilai aktual dan nilai prediksi untuk setiap data. Selisih antara nilai aktual dan prediksi kemudian dikuadratkan agar semua nilai positif. Setelah itu, rata-rata dari semua hasil kuadrat dihitung untuk mendapatkan gambaran *error* rata-rata. Terakhir, akar kuadrat dari rata-rata tersebut diambil untuk menghasilkan nilai RMSE.

```python
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Collaborative Filtering Model Evalutation')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()
```

![Collaborative Filtering Model Evaluation](https://raw.githubusercontent.com/fikridean/Movie-Recommendation-System/refs/heads/main/graph-collaborative_filtering_model_evaluation.png)

Berdasarkan *output* kode evaluasi diatas, dapat dilihat bahwa model dengan teknik *collaborative filtering* memberikan kinerja yang cukup baik yang dimana pada setiap *epoch*, RMSE model mengalami penurunan yang cukup signifikan dan terus menurun sampai *epoch* terakhir (10). Pada akhir *epoch*, dapat dilihat bahwa RMSE sudah cukup kecil yang dimana model yang dihasilkan sudah cukup baik. Berikut metrik yang dihasilkan:

- loss: 0.6048
- root_mean_squared_error: 0.1509
- val_loss: 0.6506
- val_root_mean_squared_error: 0.2096

Dari hasil evaluasi kedua model, dapat dilihat bahwa kedua model berhasil memberikan rekomendasi film yang sesuai dengan genre yang diberikan, sehingga kedua model layak untuk digunakan sebagai sistem rekomendasi film.

***Problem Statement 1***: **Bagaimana cara memahami dan memperoleh informasi mengenai data yang digunakan dalam pembuatan model sistem rekomendasi?**
- Proses eksplorasi dan pemrosesan *dataset* telah dilakukan dengan melakukan *exploratory data analysis* secara menyeluruh termasuk *univariate anaylysis*.
- Berhasil mencapai *goals* yang ditetapkan: Melakukan eksplorasi data awal (EDA) dan visualisasi untuk memahami *dataset*.
- Terdapat 5 kolom fitur yang digunakan untuk sistem rekomendasi ini yaitu `userId`, `movieId`, `title`, dan `genres`, dan `rating`.

***Problem Statement 2***: **Bagaimana cara membangun model sistem rekomendasi dengan menggunakan pendekatan *content-based filtering*?**
- Pemrosesan mencakup proses *data cleaning* yang dimana termasuk proses *removal duplicates and NaN data*, *removal irrelevant columns*, *handle imbalance data*, dan *text processing*. Selain itu, proses *data transformation*, *calculating cosine similarity*, dan *data transformation*.
- Berhasil mencapai *goals* yang ditetapkan: Membangun sistem rekomendasi film menggunakan pendekatan *content-based filtering*.

***Problem Statement 3***: **Bagaimana cara mengembangkan model sistem rekomendasi dengan pendekatan *collaborative filtering*?**  
- Pemrosesan mencakup proses *data cleaning* yang dimana termasuk proses *data cleaning* yang dimana termasuk proses *removal duplicates and NaN data*, *removal irrelevant columns*, *handle imbalance data*, dan *text processing*. Selain itu, proses *data encoding*, *data transformation*, dan *train test split* serta pelatihan model.
- Berhasil mencapai *goals* yang ditetapkan: Membangun sistem rekomendasi film menggunakan pendekatan *collaborative filtering*.

***Problem Statement 4***: **Bagaimana cara menilai kinerja model sistem rekomendasi yang telah dikembangkan?**
- Evaluasi terhadap Kedua model sistem rekomendasi dengan pendekatan *content-based filtering* dan *collaborative filtering* telah berhasil dilakukan.
- Hasil evaluasi menunjukkan bahwa kedua model menghasilkan rekomendasi film yang sesuai dengan genre yang diberikan.
- Berhasil mencapai *goals* yang ditetapkan: Mengevaluasi kinerja model sistem rekomendasi yang telah dibuat.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi berbasis *Machine Learning* yang cerdas dan relevan untuk platform *streaming online*. Sistem ini dirancang untuk mempersonalisasi pengalaman pengguna dengan merekomendasikan konten sesuai preferensi mereka, seperti film atau serial dengan genre serupa atau yang relevan dengan riwayat tontonan mereka. Pendekatan ini tidak hanya meningkatkan kepuasan pengguna tetapi juga membantu platform *streaming* dalam mempertahankan pelanggan dan meningkatkan pendapatan secara berkelanjutan.

**Dampak Proyek**

Platform *streaming online* menghadapi tantangan besar dalam menyediakan rekomendasi konten yang efektif untuk memenuhi kebutuhan pengguna. Sistem rekomendasi yang dikembangkan melalui proyek ini memberikan solusi signifikan untuk meningkatkan pengalaman pengguna dengan konten yang relevan. Selain itu, solusi ini membantu platform *streaming* meningkatkan keterlibatan pengguna, mempertahankan pelanggan, dan mendukung keberlanjutan bisnisnya, mengingat hingga 40% pendapatan platform berasal dari sistem rekomendasi.


## Reference
Jena, A. (2022). Role of a Movie Recommender System in the Streaming Industry - Muvi One. https://www.muvi.com/blogs/movie-recommender-system/

Muhd Shukri, F., Hamid, H., Nik Harun, N. A., Hamadun, F., Omran Zailuddin, M. F. N., Ariffin, Md. A., & Marzuki, I. N. (2024). The Impact of Online Streaming Platforms on the Film Industry and Film-Viewing Culture in Malaysia. Journal of Entrepreneurship and Business, 12(1), 101–115. https://doi.org/10.17687/jeb.v12i1.1194

Sekartaji, S. (2023). THE FOUR FACTORS DOMINATING ONLINE STREAMING PLATFORMS AS WITNESSED BY NETFLIX IN THE WAKE OF GLOBALIZATION. Rubikon : Journal of Transnational American Studies, 10(2), 161. https://doi.org/10.22146/rubikon.v10i2.86392
