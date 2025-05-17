# Laporan Proyek Machine Learning - Pragipta Septyaningrum Larasati

## Domain Proyek : Kesehatan
![Anemia Image](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Anemia_Image.jpg)
Anemia adalah kondisi medis yang umum terjadi di seluruh dunia, terutama pada wanita dan anak-anak, yang ditandai dengan penurunan jumlah sel darah merah atau kadar hemoglobin dalam darah. Menurut data dari Organisasi Kesehatan Dunia (WHO), sekitar 30,7% wanita usia 15–49 tahun mengalami anemia pada tahun 2023, dengan prevalensi yang lebih tinggi di antara wanita hamil. Anemia menyebabkan berkurangnya kemampuan darah untuk mengangkut oksigen ke seluruh tubuh, yang berpotensi menurunkan kualitas hidup dan meningkatkan risiko komplikasi medis seperti penyakit jantung. Selain itu, anemia pada anak-anak di usia 6-59 bulan juga cukup tinggi, yaitu sekitar 39,8% secara global[[1]](https://www.who.int/data/gho/data/themes/topics/anaemia_in_women_and_children). 

Tradisionalnya, diagnosis anemia dilakukan melalui tes laboratorium yang mengukur kadar hemoglobin dan parameter lainnya. Namun, tes laboratorium tersebut sering kali memerlukan biaya yang tinggi dan waktu yang lama untuk mendapatkan hasil. Oleh karena itu, terdapat kebutuhan untuk metode alternatif yang lebih cepat dan lebih terjangkau. Salah satu solusi yang dapat diterapkan adalah menggunakan machine learning untuk mengklasifikasikan status anemia, yang dapat dilakukan dengan memanfaatkan data medis dasar seperti kadar hemoglobin, MCV (Mean Corpuscular Volume), MCH (Mean Corpuscular Hemoglobin), dan MCHC (Mean Corpuscular Hemoglobin Concentration)[[2]](https://journal3.uin-alauddin.ac.id/index.php/msa/article/download/45083/19169/).

Dengan menggunakan algoritma klasifikasi machine learning, kita dapat membangun model yang dapat memprediksi status anemia dengan menggunakan data yang lebih sederhana, cepat, dan biaya lebih rendah. Beberapa algoritma machine learning telah terbukti efektif dalam memprediksi anemia menggunakan data medis yang tersedia. Ini tidak hanya mempercepat proses diagnosis, tetapi juga dapat membantu mengidentifikasi individu yang membutuhkan perhatian medis lebih cepat, dengan tujuan memberikan pengobatan yang lebih segera untuk mencegah komplikasi lebih lanjut[[3]](https://journal.uniku.ac.id/index.php/JESMath/article/view/11694/4941).

## Business Understanding

### Problem Statements
Adapun rumusan masalah dari analisis ini yaitu sebagai berikut.
1. Bagaimana cara melakukan prediksi status anemia pada individu menggunakan data medis dasar seperti kadar hemoglobin, MCV, MCH, dan MCHC, tanpa bergantung pada tes laboratorium yang mahal dan memakan waktu?
2. Di antara berbagai fitur yang tersedia, fitur mana yang memiliki pengaruh terbesar dalam mendeteksi anemia dengan akurat?
3. Bagaimana cara untuk mengevaluasi dan meningkatkan kinerja model yang digunakan dalam mendeteksi anemia?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan, maka diperoleh tujuan sebagai berikut.
1. Membangun model machine learning yang dapat memprediksi status anemia dengan akurat berdasarkan fitur-fitur medis dasar yang ada.
2. Mengetahui fitur-fitur yang paling berpengaruh dalam mendeteksi anemia, seperti kadar hemoglobin, MCV, MCH, dan MCHC.
3. Menggunakan evaluasi model untuk meningkatkan kinerja prediksi anemia dan mengidentifikasi area yang memerlukan perbaikan.

### Solution statements
Berdasarkan permasalahan yang ada, maka dapat diambil solusi sebagai berikut.
1. Membangun model klasifikasi dengan algoritma machine learning yang dapat memprediksi status anemia secara akurat, menggunakan fitur-fitur seperti kadar hemoglobin, MCV, MCH, dan MCHC sebagai input.
2. Melakukan analisis data untuk memahami fitur-fitur medis dasar yang berhubungan dengan status anemia, menggunakan teknik visualisasi dan korelasi untuk menggali hubungan antar fitur dan status anemia.
3. Evaluasi dan peningkatan model dengan menggunakan metrik seperti accuracy, precision, recall, dan F1-score, serta melakukan hyperparameter tuning untuk meningkatkan kinerja model dalam mendeteksi anemia dengan akurasi tinggi.

## Data Understanding

| Jenis | Keterangan |
| ------ | ------ |
| Title | [Anemia Dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset) |
| Source | [Kaggle](https://www.kaggle.com) |
| Maintainer | [Biswa Ranjan Rao](https://www.kaggle.com/biswaranjanrao) |
| License | Unknown |
| Visibility | Publik |
| Tags | Health Conditions |
| Usability | 7.06 |

Dataset yang digunakan dalam proyek ini adalah [Anemia Dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset) yang bersumber dari Kaggle. Dataset ini berisi informasi mengenai berbagai faktor medis yang digunakan untuk mendeteksi status anemia pada individu. Dataset ini mencakup fitur-fitur penting seperti kadar Hemoglobin, MCV (Mean Corpuscular Volume), MCH (Mean Corpuscular Hemoglobin), MCHC (Mean Corpuscular Hemoglobin Concentration), serta informasi tentang Gender dan status Result yang menunjukkan apakah individu tersebut mengalami anemia atau tidak anemia.

### Variabel-variabel pada Anemia dataset:
| # | Column | Dtype |
| ------ | ------ | ------ |
| 0 | Gender | int64 |
| 1 | Hemoglobin | float64 |
| 2 | MCH | float64 |
| 3 | MCHC | float64 |
| 4 | MCV | float64 |
| 5 | Result | int64 |
- Gender : merupakan jenis kelamin individu (0 = Laki-laki, 1 = Perempuan).
- Hemoglobin : merupakan kadar hemoglobin (protein) dalam sel darah merah.
- MCH : *Mean Corpuscular Hemoglobin* merupakan jumlah rata-rata hemoglobin di dalam satu sel darah merah.
- MCHC : *Mean Corpuscular Hemoglobin Concentration* merupakan konsentrasi rata-rata hemoglobin dalam satu sel darah merah.
- MCV : *Mean Corpuscular Volume* merupakan volume rata-rata sel darah merah.
- Results : merupakan label yang menunjukkan individu menderita anemia atau tidak (0 = Tidak anemia, 1 = Anemia), result adalah fitur target.

Semua kolom dalam dataset memiliki tipe data numerik, dengan empat fitur bertipe float64 (Hemoglobin, MCH, MCHC, dan MCV) dan dua fitur bertipe int64 (Gender dan Result). Hal ini menunjukkan bahwa setiap kolom sudah memiliki tipe data yang tepat. Tidak diperlukan proses encoding untuk pelatihan karena semua fitur bersifat numerik. Namun, selama tahap Exploratory Data Analysis (EDA), kolom Result dan Gender akan sementara diubah menjadi kategorik untuk mempermudah analisis visualisasi, dan setelahnya, akan dikembalikan ke tipe data semula.

![Statistik Deskriptif Image](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Statistik_Deskriptif.jpg)

Hasil pengecekan deskripsi statistik menunjukkan bahwa distribusi pada kolom Gender dan Result cukup seimbang, sedangkan untuk kolom numerik seperti Hemoglobin, MCH, MCHC, dan MCV menunjukkan variasi yang cukup signifikan. Variasi yang besar pada kolom Hemoglobin, MCH, MCHC, dan MCV adalah hal yang normal karena perbedaan kondisi antara individu yang menderita anemia dan yang tidak.

### Pengecekan Missing Value

 Langkah pertama adalah memeriksa adanya nilai yang hilang (missing values) pada setiap fitur. Apabila ditemukan missing values pada fitur yang penting, imputasi akan dilakukan menggunakan nilai median atau rata-rata (untuk fitur numerik). Jika jumlahnya sangat kecil, baris dengan missing values dapat dihapus tanpa mempengaruhi kualitas dataset secara signifikan. Penanganan missing values sangat penting karena data yang hilang dapat mengurangi kualitas model dan menyebabkan bias dalam hasil prediksi. Dengan melakukan imputasi atau penghapusan missing values, dataset menjadi lebih konsisten dan memungkinkan model untuk belajar dengan lebih efektif.
    
 Pengecekan missing values dapat menggunakan kode sebagai berikut.
  ```python
  # Memeriksa missing value
  df.isnull().sum()
  ```
  ![Missing Value](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Missing_Value.jpg)

  Pada dataset ini tidak ditemukan adanya nilai yang hilang (missing values), sehingga penanganan terhadap missing value tidak diperlukan.
  
### Pengecekan Data Duplikat

  Langkah pengecekan data duplikat bertujuan untuk memastikan tidak ada baris data yang terulang dalam dataset. Data duplikat dapat terjadi akibat kesalahan dalam proses pengumpulan atau input data, yang dapat mempengaruhi kualitas analisis dan pelatihan model. Baris duplikat yang tidak dihapus dapat memberikan bobot yang tidak tepat pada informasi yang sama, berpotensi menyebabkan overfitting atau kesalahan dalam prediksi.

  Pengecekan data duplikat dapat menggunakan kode sebagai berikut.
  ```python
  # Memeriksa duplikasi data
  jumlah_duplikat = df.duplicated().sum()
    print(f"Jumlah baris duplikat: {jumlah_duplikat}")
  ```
  Pada dataset ini ditemukan 887 baris duplikat, sehingga perlu dihapus.
  
### Pengecekan Outlier

  Pengecekan outlier dilakukan untuk mengidentifikasi nilai-nilai ekstrem yang berada jauh di luar distribusi normal data. Outlier dapat mempengaruhi kualitas model dengan memberikan kontribusi yang tidak proporsional terhadap hasil prediksi, yang sering kali mengarah pada overfitting atau kesalahan dalam model. Dalam analisis ini, outlier dapat ditemukan menggunakan metode seperti boxplot atau Interquartile Range (IQR), yang mengidentifikasi nilai-nilai yang terletak di luar batas bawah dan atas distribusi data. Penanganan outlier bisa berupa penyesuaian nilai atau penghapusan data, tergantung pada konteks dan seberapa besar pengaruhnya terhadap hasil analisis. 

  ![Outlier](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Outlier.jpg)

  Pada dataset ini tidak ada outlier yang terdeteksi pada semua kolom dalam dataset, yang menunjukkan bahwa data relatif terdistribusi secara normal tanpa adanya nilai ekstrem yang perlu ditangani..

## Exploratory Data Analysis
### Univariate Analysis
- **Analisis Distribusi Data Kategorik**
  
  ![Analisis Distribusi Fitur Kategorik](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Distribusi_Fitur_Kategorik.jpg)

  - Sebelum SMOTE, dataset menunjukkan ketidakseimbangan antara kelas Anemia dan tidak Anemia, dengan jumlah yang tidak Anemia lebih banyak.
  - Setelah SMOTE, kelas Anemia dan tidak Anemia menjadi lebih seimbang, karena SMOTE menambahkan data sintetis untuk kelas Anemia, memastikan bahwa jumlah masing-masing kelas hampir sama.
- **Analisis Distribusi Data Numerik**
  
  ![Analisis Distribusi Fitur Numerik](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Distribusi_Fitur_Numerik.jpg)

  - Distribusi Hemoglobin menunjukkan bahwa sebagian besar individu memiliki kadar hemoglobin antara 10 hingga 14 g/dL, dengan puncak frekuensi pada nilai sekitar 12 g/dL.
  - Distribusi MCH menunjukkan distribusi yang relatif merata, dengan puncak pada nilai sekitar 23-25 pg, yang mencerminkan variasi dalam ukuran rata-rata hemoglobin dalam sel darah merah.
  - Distribusi MCHC menunjukkan bahwa sebagian besar individu memiliki konsentrasi hemoglobin dalam sel darah merah yang relatif seragam, dengan puncak pada nilai sekitar 30-31 g/dL.
  - Distribusi MCV menunjukkan variasi moderat dalam volume rata-rata sel darah merah, dengan puncak frekuensi pada nilai sekitar 85-90 fL.

### Bivariate Analysis
  
  ![Analisis Distribusi Pairplot](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Pairplot.jpg)

  - Hubungan Hemoglobin dan Result: Individu dengan hemoglobin rendah lebih cenderung mengalami anemia, yang tercermin dalam distribusi yang lebih rendah untuk kelas anemia.
  - Hubungan MCH dan Result: MCH (Mean Corpuscular Hemoglobin) yang lebih rendah cenderung berhubungan dengan individu yang menderita anemia, sementara yang memiliki MCH lebih tinggi lebih cenderung tidak anemia.
  - Hubungan MCV dan Result: MCV (Mean Corpuscular Volume) yang lebih rendah juga menunjukkan kecenderungan untuk mendiagnosis anemia, dengan distribusi lebih tinggi pada kelas anemia.
  - Hubungan MCHC dan Result: MCHC (Mean Corpuscular Hemoglobin Concentration) yang lebih rendah berkorelasi dengan status anemia, menunjukkan bahwa individu dengan kadar hemoglobin lebih rendah dalam sel darah merah lebih sering menderita anemia. 

### Multivarate Analysis
- **Analisis Distribusi**
  
  ![Analisis Distribusi BoxPlot](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Boxplot.jpg)

  - Distribusi Hemoglobin Berdasarkan Gender dan Status Anemia menunjukkan perbedaan yang jelas antara individu yang tidak anemia dan anemia.
  - Wanita yang tidak anemia memiliki nilai hemoglobin lebih tinggi dengan rentang yang lebih luas, sementara wanita anemia memiliki nilai hemoglobin yang lebih rendah, dengan beberapa outlier di bawah 10 g/dL.
  - Pria yang tidak anemia juga menunjukkan nilai hemoglobin lebih tinggi, tetapi nilai hemoglobin untuk pria anemia lebih terfokus pada nilai lebih rendah, meskipun tidak sebanyak pada wanita.
  - Secara keseluruhan, individu dengan hemoglobin rendah cenderung lebih banyak berada dalam kategori anemia, dengan perbedaan yang cukup signifikan antara jenis kelamin dan status anemia.
- **Analisis Korelasi antar Fitur**
  
  ![Analisis Korelasi](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Heatmap_Korelasi.jpg)

   - Korelasi antara Hemoglobin dan Result sangat kuat dan negatif (-0.80), yang menunjukkan bahwa penurunan kadar hemoglobin sangat berhubungan dengan status anemia (Result = 1). Ini menunjukkan bahwa semakin rendah kadar hemoglobin, semakin besar kemungkinan seseorang menderita anemia.
   - Gender memiliki korelasi positif kecil dengan Result (0.25), yang menunjukkan bahwa ada sedikit hubungan antara jenis kelamin dan kemungkinan seseorang mengalami anemia, meskipun hubungan ini tidak terlalu kuat.
   - MCV dan MCHC memiliki korelasi positif sangat rendah dengan Result, menunjukkan bahwa meskipun kedua fitur ini memiliki sedikit pengaruh, mereka tidak begitu signifikan dalam memprediksi anemia jika dibandingkan dengan hemoglobin.
   - MCV dan MCH memiliki korelasi yang sangat kecil dengan fitur lainnya, menunjukkan bahwa fitur-fitur ini tidak saling terkait erat. 

## Data Preparation
Fitur pada Dataset Anemia sudah berbentuk numerik semua sehingga tidak perlu dilakukan Encoding. Adapun preprocessing yang dilakukan yaitu sebagai berikut.
- **Penghapusan Data Duplikat**

  Penghapusan data duplikat penting dilakukan untuk memastikan kualitas dataset. Data duplikat dapat terjadi akibat kesalahan input atau pengumpulan data, yang dapat mengarah pada overfitting atau kesalahan dalam pelatihan model. Ketika data yang sama muncul lebih dari satu kali, model cenderung memberi bobot yang tidak proporsional pada informasi tersebut, yang dapat mengurangi akurasi dan generalisasi model. Dengan menghapus data duplikat, kita memastikan bahwa model hanya belajar dari data yang unik dan relevan, yang membantu menghasilkan prediksi yang lebih akurat dan efisien.
  
   Untuk penghapusan data duplikat, kode berikut digunakan:
    ```python
    # Menghapus baris duplikat
    df = df.drop_duplicates()
    ```

  Sisa data setelah pembersihan baris duplikat adalah 534. Data yang terduplikasi memang cukup banyak, tetapi sisa data yang bersih sebanyak 534 (di atas 500) masih bisa untuk digunakan.
- **Data Splitting**

  Tahap ini membagi dataset menjadi data latih (80%) dan data uji (20%) menggunakan fungsi train_test_split dari sklearn. Parameter random_state=42 memastikan hasil pembagian data selalu konsisten. Data latih digunakan untuk melatih model, sedangkan data uji dipakai untuk mengevaluasi performa model pada data baru.
  
  Adapun jumlah data setelah dilakukan splitting yaitu sebagai berikut.

  | Data              | Jumlah |
  |-------------------|--------|
  | Data Keseluruhan  | 534    |
  | Data Train        | 427    |
  | Data Test         | 107    |

- **Penanganan Imbalanced Classes**

  Dengan adanya ketidakseimbangan kelas pada variabel target Result (di mana lebih banyak individu yang tidak menderita anemia), teknik seperti SMOTE (Synthetic Minority Over-sampling Technique) atau undersampling dapat diterapkan untuk memastikan model tidak terlalu condong pada kelas mayoritas (Not Anemic). Dalam hal ini, SMOTE digunakan untuk menambah jumlah sampel dari kelas Anemic. Ketidakseimbangan kelas dapat membuat model lebih cenderung memprediksi kelas mayoritas, sehingga mengabaikan kelas minoritas. Oleh karena itu, penanganan ketidakseimbangan kelas sangat penting untuk menghasilkan model yang lebih adil dan akurat. Berikut adalah perbandingan data sebelum dan setelah penerapan SMOTE.
  ![Imbalance Class](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Inbalanced_Class.jpg)
- **Feature Scaling**
    
  Proses scaling dilakukan untuk menyesuaikan rentang nilai antar fitur dalam dataset, sehingga setiap fitur berada pada skala yang seragam. Tanpa melakukan scaling, fitur dengan nilai lebih besar akan lebih mempengaruhi prediksi, sementara fitur dengan nilai lebih kecil memiliki dampak yang lebih sedikit. Dalam proyek ini, standarisasi dipilih karena distribusi data yang cenderung normal, sehingga metode ini lebih tepat digunakan. StandardScaler() dari pustaka sklearn digunakan untuk melakukan standarisasi, dengan cara mengurangi setiap nilai fitur dengan nilai rata-rata fitur tersebut, lalu membaginya dengan standar deviasi. Ini memastikan bahwa semua fitur terpusat di sekitar nol dan memiliki variansi yang seragam.

  Proses standarisasi hanya diterapkan pada data latih untuk menghindari kebocoran informasi. Adapun hasil standarisasi data yang diterapkan yaitu sebagai berikut.
  ![Standarisasi Image](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Standardisasi.jpg)

## Model Development

Pada tahap ini, beberapa algoritma machine learning digunakan untuk memecahkan masalah klasifikasi anemia, yaitu **Random Forest (RF)**, **Decision Trees (DT)**, **Logistic Regression (LR)**, **K-Nearest Neighbors (KNN)**, dan **Support Vector Machine (SVM)**. 

Pada tahap awal, model-model dasar seperti RandomForestClassifier, DecisionTreeClassifier, LogisticRegression, KNeighborsClassifier, dan SVC dilatih menggunakan parameter default dari masing-masing model. Pelatihan ini dilakukan pada data training yang telah diskalakan dan di-resample (X_train_scaled dan y_train_resampled), tanpa melakukan modifikasi pada parameter model. Tujuan dari langkah ini adalah untuk mendapatkan kinerja awal (baseline performance) dari model tanpa adanya optimasi parameter.

Setelah model-model tersebut dilatih, langkah selanjutnya adalah melakukan hyperparameter tuning untuk mencari kombinasi parameter terbaik yang dapat meningkatkan kinerja model. Proses tuning ini dilakukan dengan menggunakan GridSearchCV, yang mengeksplorasi berbagai kombinasi nilai parameter untuk setiap model, termasuk SVM, dan memilih parameter yang memberikan hasil terbaik.

Setelah parameter terbaik ditemukan, model akan dibangun kembali menggunakan kombinasi parameter terbaik tersebut dan kemudian diuji untuk memastikan apakah ada peningkatan kinerja dibandingkan dengan model yang menggunakan parameter default.

Adapun penjelasan dari model yang digunakan yaitu sebagai berikut.

### 1. **Decision Trees (DT)**
![DT Image](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Decision_Tree.jpg)

**Decision Tree** adalah algoritma machine learning yang digunakan untuk klasifikasi dan regresi. Model ini membangun pohon keputusan dengan simpul yang mewakili fitur dan cabang yang mewakili keputusan berdasarkan nilai fitur tersebut. Proses ini membuat model yang intuitif dan mudah dipahami.

Berikut adalah kode pelatihan model.

```python
dt = DecisionTreeClassifier().fit(X_train_scaled, y_train_resampled)
```

**Tahapan:**

- Pemilihan Fitur Pembagi: Memilih fitur terbaik untuk membagi data berdasarkan kriteria seperti Gini Impurity atau Entropy.
- Pembagian Data: Membagi data menjadi kelompok berdasarkan fitur yang dipilih.
- Rekursi: Mengulang pembagian hingga mencapai kriteria penghentian, seperti kedalaman maksimum pohon.
- Prediksi: Menggunakan pohon untuk mengklasifikasikan atau memprediksi data baru [[4]](https://scikit-learn.org/stable/modules/tree.html).

**Parameter yang Digunakan:**

- `max_depth`: Membatasi kedalaman pohon untuk mencegah overfitting.
- `min_samples_split`: Jumlah minimum sampel untuk membagi node.
- `min_samples_leaf`: Jumlah minimum sampel di setiap daun pohon.
- `criterion`: Kriteria untuk memilih pembagian node, seperti "gini" atau "entropy".

**Kelebihan:**

- Mudah Dipahami: Visualisasi pohon keputusan mudah untuk diinterpretasikan.
- Dapat Menangani Data Kategorikal dan Numerik: Tidak perlu transformasi fitur.
- Tidak Memerlukan Pra-pemrosesan yang Rumit: Tidak membutuhkan standarisasi atau normalisasi.

**Kekurangan:**

- Rentan terhadap Overfitting: Dapat menghasilkan model yang terlalu spesifik terhadap data pelatihan.
- Instabilitas: Sensitif terhadap perubahan kecil dalam data.
- Kinerja Menurun pada Data Tidak Seimbang: Lebih cenderung memprediksi kelas mayoritas.

### 2. **Random Forest (RF)**
![RF Image](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Decision_Tree.jpg)

**Random Forest** adalah algoritma ensemble yang membangun banyak pohon keputusan secara acak dan menggabungkan hasilnya untuk menghasilkan prediksi yang lebih akurat. Proses ini meningkatkan akurasi dengan mengurangi risiko overfitting. 

Berikut adalah kode pelatihan model.

```python
rf = RandomForestClassifier().fit(X_train_scaled, y_train_resampled)
```

**Tahapan:**

- Pembangunan Pohon Keputusan: Setiap pohon dilatih menggunakan subset data yang berbeda.
- Voting: Semua pohon memberikan suara, dan prediksi akhir dipilih berdasarkan mayoritas suara.
- Prediksi: Menggunakan hasil voting dari semua pohon untuk prediksi akhir [[5]](https://scikit-learn.org/stable/modules/ensemble.html#random-forest).

**Parameter yang Digunakan:**

- `n_estimators`: Jumlah pohon dalam hutan.
- `max_depth`: Kedalaman maksimum pohon.
- `min_samples_split`: Jumlah minimum sampel untuk membagi node.
- `min_samples_leaf`: Jumlah minimum sampel di daun pohon.

**Kelebihan:**

- Mengurangi risiko **overfitting** dibandingkan pohon keputusan tunggal.
- Cocok untuk dataset besar dan dapat menangani data yang tidak linier.
- Memberikan nilai **feature importance** yang berguna untuk analisis lebih lanjut.

**Kekurangan:**

- Proses pelatihan lebih lama jika jumlah pohon sangat besar.
- Kurang interpretatif dibandingkan dengan model pohon keputusan tunggal.

### 3. **Logistic Regression (LR)**
![LR Image](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Logistic_Regression.jpg)

**Logistic Regression** adalah model linier yang digunakan untuk klasifikasi biner. Model ini memodelkan probabilitas kelas target menggunakan fungsi logistik, yang menghasilkan nilai antara 0 dan 1 untuk klasifikasi.

Berikut adalah kode pelatihan model.

```python
lr = LogisticRegression().fit(X_train_scaled, y_train_resampled)
```

**Tahapan:**

- Kombinasi Linier: Menghitung kombinasi linier dari fitur yang ada.
- Fungsi Logistik: Fungsi logistik digunakan untuk mengubah nilai linier menjadi probabilitas.
- Prediksi: Mengklasifikasikan data berdasarkan probabilitas yang dihasilkan oleh fungsi logistik [[6]](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

**Parameter yang Digunakan:**

- `penalty`: Jenis regulasi yang digunakan untuk menghindari overfitting (L1 atau L2).
- `C`: Parameter kontrol regulasi, mengontrol kekuatan regulasi.
- `solver`: Algoritma untuk optimisasi (misalnya "liblinear", "saga").

**Kelebihan:**

- Cepat dan efisien untuk dataset besar dengan fitur linier.
- Menyediakan probabilitas untuk hasil, yang berguna dalam analisis lebih lanjut.
- Sederhana dan mudah diimplementasikan.

**Kekurangan:**

- Tidak cocok untuk data dengan hubungan non-linier yang kompleks.
- Kinerja dapat menurun jika fitur tidak terstandarisasi dengan baik.

### 4. **K-Nearest Neighbors (KNN)**
![KNN Image](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/KNN.jpg)

**K-Nearest Neighbors (KNN)** adalah algoritma non-parametrik yang mengklasifikasikan data berdasarkan kedekatannya dengan data lainnya. Algoritma ini menggunakan k tetangga terdekat untuk menentukan kelas atau nilai target.

Berikut adalah kode pelatihan model.

```python
knn = KNeighborsClassifier().fit(X_train_scaled, y_train_resampled)
```

**Tahapan:**

- Menghitung Jarak: Menghitung jarak antara data yang ingin diprediksi dan data lain di training set menggunakan metrik jarak seperti Euclidean Distance.
- Pemilihan Tetangga: Memilih k tetangga terdekat berdasarkan jarak yang dihitung.
- Klasifikasi: Mengklasifikasikan data berdasarkan mayoritas kelas dari k tetangga terdekat [[7]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

  
**Parameter yang Digunakan:**

- `n_neighbors`: Jumlah tetangga terdekat yang digunakan untuk klasifikasi.
- `metric`: Metrik jarak yang digunakan untuk menghitung kedekatan (misalnya, "euclidean").
- `weights`: Metode pembobotan tetangga (uniform atau distance).

**Kelebihan:**

- Sederhana dan mudah dipahami.
- Tidak memerlukan model eksplisit; hanya memerlukan data untuk melakukan prediksi.
- Sangat baik untuk masalah dengan data tidak terstruktur.

**Kekurangan:**

- Proses prediksi sangat lambat pada dataset besar, karena harus menghitung jarak ke semua titik data.
- Rentan terhadap data yang berisik (noisy data) dan tidak efektif pada data dengan dimensi tinggi.

### 5. **Suport Vector Machine (SVM)**
![SVM Image](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/SVM.jpg)

**Suport Vector Machine (SVM)** adalah algoritma pembelajaran mesin yang digunakan untuk klasifikasi dan regresi. SVM berfungsi dengan mencari hyperplane terbaik yang dapat memisahkan data dari kelas yang berbeda.

Berikut adalah kode pelatihan model.

```python
knn = KNeighborsClassifier().fit(X_train_scaled, y_train_resampled)
```

**Tahapan:**

- Pemilihan Hyperplane: Menentukan garis atau hyperplane terbaik untuk memisahkan kelas yang berbeda.
- Maximizing Margin: Memaksimalkan margin antara hyperplane dan data dari kelas yang berbeda.
- Prediksi: Menggunakan hyperplane untuk memprediksi kelas data baru. [[8]](https://scikit-learn.org/stable/modules/svm.html).

  
**Parameter yang Digunakan:**

- `C`: Parameter yang mengontrol regulasi, mengatur margin dan kesalahan klasifikasi.
- `kernel`: Fungsi kernel yang digunakan untuk memetakan data ke ruang dimensi lebih tinggi (linear, polynomial, RBF, dll).
- `gamma`: Parameter untuk kernel yang mengontrol bentuk kurva keputusan.

**Kelebihan:**

- Baik untuk data dengan dimensi tinggi dan data non-linier.
- Dapat menangani data yang tidak terstruktur dengan menggunakan kernel trick.

**Kekurangan:**

- Proses pelatihan yang lebih lama pada dataset besar.
- Membutuhkan pemilihan parameter yang cermat untuk mencapai performa terbaik.

### Hyperparameter Tuning

Hyperparameter Tuning bertujuan untuk mengoptimalkan kinerja model dengan mencari kombinasi parameter terbaik yang dapat meningkatkan akurasi dan efisiensi prediksi. Setiap algoritma machine learning memiliki hyperparameter yang dapat disesuaikan, seperti jumlah pohon dalam Random Forest, kedalaman maksimum pada Decision Tree, atau nilai C dan gamma pada SVM. Dengan melakukan tuning pada hyperparameter ini, model dapat beradaptasi lebih baik dengan data dan menghasilkan prediksi yang lebih akurat. Hyperparameter Tuning yang dilakukan dalam analisis ini yaitu GridSearchCV.

Contoh kode tuning untuk **Decision Tree**:

```python
# GridSearch untuk Decision Tree
dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeClassifier(random_state=42)
grid_dt = GridSearchCV(dt, dt_params, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_dt.fit(X_train_scaled, y_train_resampled)

print("Best parameters (DT):", grid_dt.best_params_)
```
  
### Pemilihan Model Terbaik

Random Forest (RF) adalah algoritma ensemble yang efektif untuk menangani dataset dengan ukuran terbatas, seperti dataset anemia yang berjumlah 534 data, karena kemampuannya mengurangi risiko overfitting dengan membangun beberapa pohon keputusan yang beragam dan menggabungkan hasilnya, sehingga menghasilkan model yang lebih stabil dan akurat. Keunggulan lainnya adalah kemampuan untuk menangani hubungan non-linier antar fitur dan memberikan informasi tentang pentingnya setiap fitur, yang berguna untuk analisis lebih lanjut. Selain itu, Random Forest tidak memerlukan standarisasi fitur dan dapat menangani data yang hilang tanpa mempengaruhi hasil secara signifikan. Meskipun demikian, perlu dilakukan evaluasi performa menggunakan metrik seperti akurasi, presisi, recall, dan F1-score untuk memastikan bahwa model tetap optimal dan dapat bekerja dengan baik pada data yang ada.

## Evaluation

Pada tahap ini, metrik evaluasi yang digunakan untuk mengukur performa model meliputi **Akurasi**, **Precision**, **Recall**, dan **F1-Score**. Metrik-metrit ini dipilih karena relevansi mereka dalam konteks masalah klasifikasi biner yang ada pada proyek ini, yaitu memprediksi apakah seorang individu menderita anemia atau tidak.

**1. Akurasi**

  Akurasi adalah metrik yang mengukur seberapa banyak prediksi yang benar dibandingkan dengan total prediksi yang dilakukan. Dalam kasus klasifikasi biner, akurasi dihitung dengan rumus:

  $$
  \text{Akurasi} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Observations}}
  $$

  Di mana **True Positives (TP)** adalah jumlah individu yang benar-benar menderita anemia dan diprediksi menderita anemia, sedangkan **True Negatives (TN)** adalah jumlah individu yang tidak menderita anemia dan diprediksi tidak menderita anemia. Akurasi yang tinggi menunjukkan bahwa model berhasil memprediksi dengan benar sebagian besar data.

**2. Precision**

  Precision mengukur seberapa banyak prediksi positif yang benar (yaitu, individu yang diprediksi menderita anemia dan benar-benar menderita anemia) dibandingkan dengan seluruh prediksi positif yang dibuat oleh model. Formula precision adalah:

  $$
  \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
  $$

  Precision tinggi menunjukkan bahwa model sangat berhati-hati dalam mengklasifikasikan individu sebagai menderita anemia dan memiliki lebih sedikit kesalahan klasifikasi (false positives). Precision yang tinggi berarti model dapat mengidentifikasi individu yang benar-benar menderita anemia dengan baik, menghindari prediksi yang salah terhadap individu yang sehat. Ini sangat penting ketika tujuan adalah meminimalkan **false positives**, misalnya, untuk menghindari pemberian diagnosis yang salah.

#### 3. **Recall**

  Recall mengukur kemampuan model dalam menemukan semua kasus positif yang sebenarnya (yaitu, mendeteksi semua individu yang benar-benar menderita anemia). Formula recall adalah:

  $$
  \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
  $$

  Recall tinggi menunjukkan bahwa model berhasil menangkap sebagian besar individu yang benar-benar menderita anemia, meskipun mungkin ada beberapa kesalahan (false negatives). Recall yang tinggi sangat diinginkan dalam kasus diagnosis medis, karena **lebih penting** untuk **menangkap semua pasien yang menderita anemia** (mencegah **false negatives**) daripada menghindari beberapa **false positives**.

#### 4. **F1-Score**

  F1-Score adalah rata-rata harmonis antara **Precision** dan **Recall**. F1-Score memberikan keseimbangan antara precision dan recall, yang berguna ketika kita menginginkan performa yang baik dalam kedua aspek tersebut. Formula F1-Score adalah:

  $$
  \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

  F1-Score yang tinggi menunjukkan bahwa model memberikan keseimbangan yang baik antara ketepatan prediksi dan kemampuan untuk mendeteksi semua kasus positif. F1-Score yang baik menunjukkan bahwa model tidak hanya akurat dalam memprediksi anemia tetapi juga berhasil mendeteksi sebagian besar individu yang benar-benar menderita anemia, yang sangat penting dalam konteks kesehatan.

### Hasil Proyek Berdasarkan Metrik Evaluasi
Setelah melakukan pelatihan dan evaluasi model menggunakan **cross-validation**, hasil yang didapatkan menunjukkan performa model yang beragam tergantung pada algoritma yang digunakan. Terlihat bahwa performa model setelah dilakukan hyperparameter tuning memberikan hasil yang sedikit lebih baik dari sebelum melakukan hyperparameter tuning. 

![Perbandingan Evaluasi Image](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Perbandingan.jpg)

Adapun ringkasan metrik evaluasi untuk model yang diuji **sebelum dilakukan hyperparameter tuning** yaitu sebagai berikut.

![Evaluasi Before Image](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Metrik_Evaluasi_Sebelum_Tuning.jpg)

- **Random Forest (RF)**:

  * **Akurasi**: 99.07%
  * **Precision**: 99.08%
  * **Recall**: 99.07%
  * **F1-Score**: 99.07%

  Random Forest memberikan hasil yang sangat baik dalam mendeteksi anemia, dengan akurasi dan recall yang tinggi. Model ini berhasil mendeteksi sebagian besar individu yang menderita anemia, sambil mempertahankan tingkat kesalahan prediksi yang rendah.

- **Decision Tree (DT)**:

  * **Akurasi**: 98.13%
  * **Precision**: 98.20%
  * **Recall**: 98.13%
  * **F1-Score**: 98.13%

  Menunjukkan bahwa model ini masih memberikan performa yang sangat baik, meskipun sedikit lebih rendah dibandingkan dengan Random Forest. Precision yang tinggi menunjukkan bahwa sebagian besar prediksi positifnya benar, namun Recall yang sedikit lebih rendah menunjukkan bahwa model ini sedikit lebih sering melewatkan individu dengan anemia

- **Logistic Regression (LR)**:

  * **Akurasi**: 96.26%
  * **Precision**: 96.54%
  * **Recall**: 96.26%
  * **F1-Score**: 96.27%
  
  Meskipun tidak sebaik Random Forest, model ini tetap mampu mengklasifikasikan data dengan baik, terutama dalam hal Precision dan Recall yang seimbang. Namun, Recall yang lebih rendah menunjukkan bahwa model ini lebih sering melewatkan individu yang menderita anemia dibandingkan dengan model lainnya
  
- **K-Nearest Neighbors (KNN)**:

  * **Akurasi**: 90.65%
  * **Precision**: 91.69%
  * **Recall**: 90.65%
  * **F1-Score**: 90.66%
    
   Meskipun memiliki Precision dan Recall yang seimbang, KNN lebih sering melewatkan individu yang menderita anemia dan memiliki Accuracy yang jauh lebih rendah dibandingkan model lainnya.

Secara keseluruhan, **Random Forest** menjadi model yang paling unggul dalam hal akurasi, recall, dan keseimbangan metrik evaluasi, diikuti oleh **Decision Tree**, **Logistic Regression**, dan **K-Nearest Neighbors**. Model ini memberikan hasil yang optimal pada dataset yang diuji.

Berikut adalah ringkasan metrik evaluasi untuk model yang diuji **setelah dilakukan hyperparameter tuning**:

  ![Evaluasi After Image](https://raw.githubusercontent.com/Pragiptalrs/Predictive-Analytics/main/Metrik_Evaluasi_Setelah_Tuning.jpg)

- **Random Forest (RF)**:

  * **Akurasi**: 99.07%
  * **Precision**: 99.08%
  * **Recall**: 99.07%
  * **F1-Score**: 99.07%

  Random Forest memberikan hasil yang sangat baik dalam mendeteksi anemia, dengan akurasi dan recall yang tinggi. Model ini berhasil mendeteksi sebagian besar individu yang menderita anemia, sambil mempertahankan tingkat kesalahan prediksi yang rendah.

- **Decision Tree (DT)**:

  * **Akurasi**: 99.07%
  * **Precision**: 99.08%
  * **Recall**: 99.07%
  * **F1-Score**: 99.07%
    
  Meskipun hasilnya hampir serupa dengan Random Forest, Decision Tree bisa lebih rentan terhadap overfitting dibandingkan dengan Random Forest karena tidak menggunakan ensemble learning, yang dapat membatasi kemampuan model untuk generalisasi pada data baru.

- **Logistic Regression (LR)**:

  * **Akurasi**: 97.20%
  * **Precision**: 97.36%
  * **Recall**: 97.20%
  * **F1-Score**: 97.20%
  
  Logistic Regression memberikan hasil solid, tetapi model ini menunjukkan kinerja yang lebih rendah dibandingkan dengan Random Forest dan Decision Tree, terutama dalam Recall, yang berarti model ini sedikit lebih sering melewatkan individu dengan anemia.
  
- **K-Nearest Neighbors (KNN)**:

  * **Akurasi**: 90.65%
  * **Precision**: 91.69%
  * **Recall**: 90.65%
  * **F1-Score**: 90.65%
    
  KNN memberikan hasil Precision dan Recall yang seimbang. Namun, KNN menunjukkan hasil yang lebih rendah dibandingkan dengan model lainnya, yang mengindikasikan bahwa KNN tidak optimal untuk dataset ini dan cenderung lebih sering melewatkan individu dengan anemia.

Secara keseluruhan, **Random Forest** tetap menjadi model yang paling disarankan setelah tuning dengan nilai F1-Score yang sangat tinggi dan metrik evaluasi lainnya yang juga tinggi, diikuti oleh Decision Tree, dengan Logistic Regression dan KNN lebih cocok digunakan dalam skenario yang lebih sederhana.

### Model Terbaik Berdasarkan Metrik Evaluasi

F1-Score yang tinggi sangat penting dalam diagnosis medis karena ini membantu menyeimbangkan dua hal utama: precision (seberapa tepat model dalam mengidentifikasi pasien yang benar-benar menderita anemia) dan recall (seberapa baik model dalam mendeteksi semua pasien yang menderita anemia). Dalam hal ini, F1-Score memberikan gambaran yang lebih lengkap tentang kinerja model, memastikan model tidak hanya akurat, tetapi juga tidak melewatkan pasien yang membutuhkan perawatan. Random Forest adalah model yang sangat baik dalam hal ini karena memberikan nilai F1-Score yang tinggi dan stabil. Meskipun Decision Tree juga memiliki performa yang baik, Random Forest lebih unggul karena menggunakan beberapa pohon keputusan yang membantu mengurangi kesalahan dan overfitting, sehingga menghasilkan model yang lebih akurat.

### **Kesimpulan**
**1. Cara Prediksi Status Anemia:**

Berdasarkan analisis, model klasifikasi machine learning seperti Decision Tree, Random Forest, dan Support Vector Machine (SVM) dapat digunakan untuk memprediksi status anemia hanya dengan menggunakan data medis dasar seperti kadar hemoglobin, MCV, MCH, dan MCHC. Model ini dapat memberikan prediksi yang akurat tanpa memerlukan tes laboratorium tambahan yang mahal dan memakan waktu. Hasil model yang diuji menunjukkan akurasi yang sangat baik, dengan model Decision Tree dan Random Forest memberikan akurasi lebih dari 97%.

**2. Fitur yang Paling Berpengaruh:**

Dari analisis yang dilakukan, kadar Hemoglobin terbukti menjadi fitur yang paling berpengaruh dalam mendeteksi anemia dengan akurat. Ini tercermin dari hasil evaluasi model yang menunjukkan hubungan yang kuat antara kadar hemoglobin dan status anemia. Fitur lainnya seperti MCV, MCH, dan MCHC juga penting, tetapi tidak sekuat hemoglobin dalam memprediksi anemia. Hemoglobin memiliki korelasi yang sangat tinggi dengan status anemia, menjadikannya fitur yang dominan dalam membedakan individu yang anemia dan yang tidak.

**3. Evaluasi dan Peningkatan Kinerja Model:**

- Kinerja model dievaluasi dengan menggunakan metrik seperti accuracy, precision, recall, dan F1-score. Berdasarkan hasil analisis:
- Decision Tree dan Random Forest menunjukkan kinerja terbaik dengan F1-Score dan accuracy yang sangat tinggi (mendekati 1.0).
- Logistic Regression dan Support Vector Machine (SVM) juga memberikan hasil yang sangat baik, meskipun sedikit lebih rendah dibandingkan Random Forest dan Decision Tree.
- K-Nearest Neighbors (KNN) menunjukkan hasil yang lebih rendah dibandingkan dengan model lainnya.
- Peningkatan kinerja model dilakukan melalui hyperparameter tuning, yang membantu mengoptimalkan parameter model dan mengurangi overfitting atau underfitting. GridSearchCV digunakan untuk menemukan kombinasi parameter terbaik, yang kemudian meningkatkan kinerja model dalam memprediksi status anemia.

## Referensi:

[[1] World Health Organization, "Anaemia in women and children," Global Health Observatory Data Repository, 2023. [Online]. Available: https://www.who.int/data/gho/data/themes/topics/anaemia_in_women_and_children. [Accessed: May 16, 2025].](https://www.who.int/data/gho/data/themes/topics/anaemia_in_women_and_children)

[[2] Ermawati, R. Ibnas, & B. A. Kurniawan, "Klasifikasi Penderita Anemia Menggunakan Metode Regresi Logistik," Jurnal Matematika dan Statistika serta Aplikasinya, vol. 11, no. 2, pp. 93-98, Jul.-Dec. 2023.](https://journal3.uin-alauddin.ac.id/index.php/msa/article/download/45083/19169/)

[[3] Zam, N. W., Irwan, I., & Irwan, M., "Klasifikasi Machine Learning untuk Anemia Menggunakan Metode Support Vector Machine dan Random Forest," Jurnal Edukasi dan Sains Matematika (JES-MAT), vol. 11, no. 1, pp. 62-76, Mar. 2025.](https://journal.uniku.ac.id/index.php/JESMath/article/view/11694/4941)

[[4]Scikit-learn documentation. "Decision Trees," Scikit-learn, 2021. [Online]. Available: https://scikit-learn.org/stable/modules/tree.html. [Accessed: 16-May-2025].](https://scikit-learn.org/stable/modules/tree.html)

[[5]Scikit-learn documentation. "Random Forest," Scikit-learn, 2021. [Online]. Available: https://scikit-learn.org/stable/modules/ensemble.html#random-forest. [Accessed: 16-May-2025].](https://scikit-learn.org/stable/modules/ensemble.html#random-forest)

[[6]Scikit-learn documentation. "LogisticRegression," Scikit-learn, 2021. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html. [Accessed: 16-May-2025].](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

[[7]Scikit-learn documentation. "KNeighborsClassifier," Scikit-learn, 2021. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html. [Accessed: 16-May-2025].](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

[[8]Scikit-learn documentation. "SupportVectorMachine," Scikit-learn, 2021. [Online]. Available: https://scikit-learn.org/stable/modules/svm.html. [Accessed: 16-May-2025].](https://scikit-learn.org/stable/modules/svm.html)
