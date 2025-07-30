!pip install transformers
!pip install datasets
!pip install stanza
!pip install Sastrawi
!pip install wordcloud

"""# Library"""

import pandas as pd
import numpy as np
import re
import string
import stanza
from collections import Counter
from tqdm import tqdm
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
import nltk
from collections import Counter
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F
from sklearn.cluster import KMeans
import torch
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from datasets import Dataset
from torch.utils.data import TensorDataset
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import  accuracy_score
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix

"""## Function"""

def create_corpus(data, var):
    corpus = []
    for x in data[var].str.split():
        for i in x:
            corpus.append(i)
    return corpus

# Get data containing specific word function
def get_data_contains_word(data, feature, string):
    _id = []
    for text, ID in zip(data[feature], data['review_id']):
        if string in text:
            _id.append(ID)
    return data[data['review_id'].isin(_id)]

# Remove Indonesian stop words using Sastrawi
def remove_indonesian_stop(data, feature):
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())
    filtered = []
    for text in data[feature]:
        text_list = [word for word in text.split() if word not in stopwords]
        filtered.append(' '.join(text_list))
    return filtered

# Plotting words containing specific word function
def docs_contain_word_plot(data, var, num_words, width, height):
    words, counts = [], []
    corpus = list(set(create_corpus(data, var)))
    for word in tqdm(corpus):
        words.append(word)
        counts.append(get_data_contains_word(data, var, word).shape[0])
    res = pd.DataFrame({'words': words, 'counts': counts}).sort_values(by='counts', ascending=False)
    plt.figure(figsize=[width, height])
    sns.barplot(data=res.iloc[:num_words], x='counts', y='words')
    for index, value in enumerate(res['counts'].iloc[:num_words]):
        plt.text(value, index, value)
    plt.show()

# Count numerical occurrences function
def num_count(data, feature):
    num_list = []
    for sentence in tqdm(data[feature]):
        num = sum(1 for word in sentence if word.isdigit())
        num_list.append(num)
    return num_list

# Count numerical occurrences function
def num_count(data, feature):
    num_list = []
    for sentence in tqdm(data[feature]):
        num = sum(1 for word in sentence if word.isdigit())
        num_list.append(num)
    return num_list

# Plot unique word count function
def plot_unique_word_count(corpus, width, height, range1, range2, title, color, ax=None):
    words, values, len_words = [], [], []
    for word, value in zip(pd.DataFrame(corpus).value_counts().index, pd.DataFrame(corpus).value_counts()):
        words.append(word[0])
        values.append(value)
        len_words.append(len(word[0]))
    res = pd.DataFrame({'words': words, 'values': values, 'len_words': len_words}).sort_values(by='values', ascending=False)
    ax = ax
    ax.set_title(title)
    sns.barplot(data=res[range1:range2], y='words', x='values', color=color, ax=ax)
    for index, value in enumerate(res['values'].iloc[range1:range2]):
        plt.text(value, index, value)
    return ax

# Generate dataframe of unique word count function
def df_unique_word_count(corpus):
    words, values, len_words = [], [], []
    for word, value in zip(pd.DataFrame(corpus).value_counts().index, pd.DataFrame(corpus).value_counts()):
        words.append(word[0])
        values.append(value)
        len_words.append(len(word[0]))
    return pd.DataFrame({'words': words, 'values': values, 'len_words': len_words}).sort_values(by='values', ascending=False)

# word cloud
def plot_word_cloud(corpus, title, ax):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(corpus)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=18, weight='bold')
    ax.axis('off')

"""## Typo dictionary"""

words_dict = {
    '1000gb': '1000 GB',
    '10gb': '10 GB',
    '20gb': '20 GB',
    '247gb': '247 GB',
    '2nd': 'kedua',
    '30mnt': '30 menit',
    '5ghz': '5 GHz',
    'Basecamp': 'kem',
    'aamiin': 'amin',
    'absen': 'ketidakhadiran',
    'absensi': 'daftar hadir',
    'acara': 'program',
    'acc': 'terima',
    'access': 'akses',
    'account': 'akun',
    'acnya': 'ac nya',
    'ad': 'ada',
    'adain': 'adakan',
    'adem': 'sejuk',
    'admin': 'administrator',
    'advance': 'tinggi',
    'afk': 'tidak hadir',
    'agak': 'sedikit',
    'airnya': 'air nya',
    'aiueo': '',
    'aja': 'saja',
    'ajar': 'mengajar',
    'ama': 'sama',
    'atass': 'atas',
    'attendence': 'absensi',
    'baikk': 'baik',
    'banget': 'sekali',
    'bangettttt': 'sekali',
    'bbrp': 'beberapa',
    'belajarr': 'belajar',
    'bener': 'benar',
    'bgs': 'bagus',
    'bgt': 'sekali',
    'bikin': 'buat',
    'bli': 'beli',
    'blibet': 'rumit',
    'blm': 'belum',
    'bolong': 'lubang',
    'bs': 'bisa',
    'byk': 'banyak',
    'camp': 'kem',
    'cape': 'lelah',
    'capek': 'lelah',
    'cas': 'mengisi',
    'cepet': 'cepat',
    'cewe': 'wanita',
    'cewek': 'wanita',
    'chek': 'cek',
    'concern': 'kecemasan',
    'connect': 'terhubung',
    'cowo': 'pria',
    'cowok': 'pria',
    'cuman': 'hanya',
    'dalem': 'dalam',
    'dapet': 'dapat',
    'dempet': 'dekat',
    'dg': 'dengan',
    'dgn': 'dengan',
    'dibenarkannn': 'dibenarkan',
    'dibenerin': 'diperbaiki',
    'dikasi': 'dikasih',
    'dikit': 'sedikit',
    'dilimit': 'dibatasi',
    'diperibet': 'dirumitkan',
    'direnov': 'di renovasi',
    'diupdate': 'diperbarui',
    'dll': 'dan lain lain',
    'dlm': 'dalam',
    'doang': 'saja',
    'dpt': 'dapat',
    'dr': 'dari',
    'dri': 'dari',
    'dtg': 'datang',
    'effort': 'usaha',
    'encok': 'nyeri',
    'eror': 'galat',
    'error': 'galat',
    'escalator': 'eskalator',
    'g': 'tidak',
    'ga': 'tidak',
    'gaada': 'tidak ada',
    'gaadaa': 'tidak ada',
    'gabisa': 'tidak bisa',
    'gada': 'tidak ada',
    'gak': 'tidak',
    'gakada': 'tidak ada',
    'ganisa': 'tidak bisa',
    'gapunya': 'tidak punya',
    'gatau': 'tidak tahu',
    'gdrive': 'google drive',
    'ged': 'gedung',
    'gede': 'besar',
    'gitu': 'begitu',
    'gk': 'tidak',
    'gkada': 'tidak ada',
    'gmana': 'bagaimana',
    'gmeet': 'google meet',
    'gmn': 'bagaimana',
    'happy': 'bahagia',
    'hrs': 'harus',
    'idup': 'hidup',
    'info': 'informasi',
    'iniiii': 'ini',
    'jadi': 'jadi',
    'jdi': 'jadi',
    'jeblok': 'turun',
    'jebol': 'terbongkar',
    'jelek': 'buruk',
    'jg': 'juga',
    'jgn': 'jangan',
    'jln': 'jalan',
    'k': 'ke',
    'kadang': 'terkadang',
    'kalo': 'kalau',
    'kamar': 'kamar',
    'karna': 'karena',
    'kasian': 'kasihan',
    'kaya': 'seperti',
    'kayak': 'seperti',
    'kbm': 'kegiatan belajar mengajar',
    'kecoak': 'kecoa',
    'kejeblos': 'terperosok',
    'kenceng': 'kencang',
    'kenchang': 'kencang',
    'kl': 'kalau',
    'klo': 'kalau',
    'kluarga': 'keluarga',
    'kmr': 'kamar',
    'knowledge': 'pengetahuan',
    'kpd': 'kepada',
    'kran': 'keran',
    'krg': 'kurang',
    'krn': 'karena',
    'krna': 'karena',
    'kyk': 'seperti',
    'lbh': 'lebih',
    'lcd': 'proyektor',
    'lemot': 'lambat',
    'lemottt': 'lambat',
    'letoy': 'tidak tegak',
    'lg': 'lagi',
    'lgi': 'lagi',
    'lgsg': 'langsung',
    'lgsung': 'langsung',
    'lt': 'lantai',
    'makasi': 'terima kasih',
    'makasih': 'terima kasih',
    'malah': 'semakin',
    'management': 'manajemen',
    'masi': 'masih',
    'matkul': 'mata kuliah',
    'max': 'maksimal',
    'medsos': 'media sosial',
    'meleyot': 'bengkok',
    'menceng': 'miring',
    'mendadak': 'tiba tiba',
    'mengupload': 'mengunggah',
    'mgkin': 'mungkin',
    'mhs': 'mahasiswa',
    'mkn': 'makan',
    'mlm': 'malam',
    'mnt': 'menit',
    'mood': 'suasana hati',
    'msh': 'masih',
    'mslh': 'masalah',
    'mslhnya': 'masalah nya',
    'n': 'dan',
    'naro': 'taruh',
    'ndak': 'tidak',
    'ngajar': 'mengajar',
    'ngajarin': 'mengajar',
    'ngajarinya': 'mengajar',
    'ngak': 'tidak',
    'ngasih': 'membagikan',
    'ngaturin': 'mengatur',
    'ngebales': 'balas',
    'ngecek': 'memeriksa',
    'ngelag': 'macet',
    'ngga': 'tidak',
    'nggak': 'tidak',
    'nglag': 'macet',
    'ngumpulin': 'mengumpulkan',
    'nnt': 'nanti',
    'non': 'bukan',
    'nunggu': 'tunggu',
    'ny': 'nya',
    'nyaaa': 'nya',
    'oke': 'setuju',
    'oncam': 'kamera hidup',
    'onlen': 'online',
    'org': 'orang',
    'pake': 'pakai',
    'panasss': 'panas',
    'pd': 'pada',
    'pdhl': 'padahal',
    'pembayan': 'pembayaran',
    'pesen': 'pesan',
    'pjj': 'pembelajaran jarak jauh',
    'poll': 'sekali',
    'ppt': 'presentasi',
    'record': 'rekaman',
    'regards': 'salam',
    'renov': 'renovasi',
    'reot': 'reyot',
    'repottt': 'repot',
    'ribet': 'rumit',
    'sampe': 'sampai',
    'samsek': 'sama sekali',
    'samsekk': 'sama sekali',
    'sangatta': 'sangat',
    'sdh': 'sudah',
    'serem': 'seram',
    'sgt': 'sangat',
    'slow': 'lambat',
    'sm': 'sama',
    'smt': 'semester',
    'sower': 'shower',
    'spt': 'seperti',
    'standart': 'standar',
    'sy': 'saya',
    'sya': 'saya',
    'tak': 'tidak',
    'tambahi': 'tambahkan',
    'tanyak': 'tanya',
    'taru': 'taruh',
    'tau': 'tahu',
    'tdk': 'tidak',
    'tdr': 'tidur',
    'tentuta': 'tentu',
    'tetep': 'tetap',
    'tgl': 'tanggal',
    'think': 'pikir',
    'tiap': 'setiap',
    'time': 'waktu',
    'tissue': 'tisu',
    'tisue': 'tisu',
    'tlg': 'tolong',
    'tlp': 'telpon',
    'tmpt': 'tempat',
    'toilettttt': 'toilet',
    'tp': 'tapi',
    'tpi': 'tapi',
    'trus': 'terus',
    'tsb': 'tersebut',
    'ttg': 'tentang',
    'ttng': 'tentang',
    'udah': 'sudah',
    'udh': 'sudah',
    'univ': 'universitas',
    'update': 'perbarui',
    'utk': 'untuk',
    'visit': 'kunjungi',
    'y': 'ya',
    'yabg': 'yang',
    'yakali': 'masa iya',
    'yg': 'yang',
    '&': 'dan'
}

"""# Load Data"""

df = pd.read_csv('/content/drive/MyDrive/Aspect Category Sentiment Analysis/Dataset/Dataset_SUYA.csv')
display(df)

df.info()

data_2 = pd.read_csv('/content/drive/MyDrive/Aspect Category Sentiment Analysis/Dataset/Dataset_SUYA_2023_all.csv')
display(data_2)

data_2.info()

data_x=pd.concat([df, data_2], ignore_index=True)
display(data_x)

data = pd.concat([df, data_2], ignore_index=True)

data.info()

data.isnull().sum()

data = data.dropna()
data.isnull().sum()

display(data[data.duplicated()])

data = data.drop_duplicates()
data.reset_index(drop=True, inplace=True)
data.info()

data = data.rename(columns={'Kategori' : 'category', 'Aspirasi' : 'opinion'})
data.info()

# category_counts = data['category'].value_counts()
# category_counts

# plt.figure(figsize=(8, 6))
# category_counts.plot(kind='bar')
# plt.title('Number of Data per Category')
# plt.xlabel('Category')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# category_percentages = (category_counts / len(data)) * 100

# print(category_counts)
# print(category_percentages)

# plt.figure(figsize = (7, 7))
# plt.pie(data['category'].value_counts(), labels = data['category'].value_counts().index, autopct = '%1.1f%%', startangle=180, counterclock=False)
# plt.legend(data['category'].value_counts().index, loc="center left", bbox_to_anchor=(1, 0.5))
# plt.title("Percentage of Data Distribution by Category", fontsize = 15, fontweight= "bold")
# plt.show()

"""# Preprocessing

## Casefolding
"""

data['opinion_cleaned'] = [i.lower() for i in data['opinion']]

display(data)

"""## Remove characters"""

data['opinion_cleaned'] = [re.sub(r'[^\x00-\x7f]',r'', i) for i in data['opinion_cleaned']]
data['opinion_cleaned'] = [re.sub(r'[^A-Za-z0-9\s]', ' ', i) for i in data['opinion_cleaned']]

display(data)

"""## Remove excess space"""

# Menghapus spasi tambahan dan \n
data['opinion_cleaned'] = [re.sub(r'\s+', ' ', i) for i in data['opinion_cleaned']]
data['opinion_cleaned'] = [re.sub(r'\n', r' ', i) for i in data['opinion_cleaned']]

display(data)

"""## Remove numbering"""

# Menghapus angka
data['opinion_cleaned'] = [re.sub(r'\d+', '', i) for i in data['opinion_cleaned']]

display(data)

"""# Normalization"""

# memperbaiki typo
list_sentence_data = []
for sentence in tqdm(data['opinion_cleaned']):
    cleaned_sentence = [words_dict[word] if word in list(words_dict.keys()) else word for word in sentence.split()]
    list_sentence_data.append(' '.join(cleaned_sentence))
data['opinion_cleaned'] = list_sentence_data

display(data)

"""# Stopword removal"""

data['opinion_cleaned_nostopwords'] = remove_indonesian_stop(data, 'opinion_cleaned')

display(data)

"""# Tokenizer"""

stanza.download('id')  # Pastikan model Bahasa Indonesia terinstal
nlp = stanza.Pipeline('id', processors='tokenize')

# Tokenization using Stanza
def stanza_tokenizer(text):
    doc = nlp(text)
    tokens = [word.text for sentence in doc.sentences for word in sentence.words]
    return tokens

data['opinion_tokenized'] = data['opinion_cleaned_nostopwords'].apply(stanza_tokenizer)

"""# Stemming"""

# Membuat objek stemmer
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

# Fungsi stemmer
def stemmer_text(text):
    return stemmer.stem(" ".join(text))

# Stemmer
data['opinion_stemmer'] = data['opinion_tokenized'].apply(stemmer_text)

"""# Feature Selection"""

data_fix = data.drop(['category', 'opinion', 'opinion_cleaned', 'opinion_cleaned_nostopwords'], axis=1)

display(data_fix)

data_fix.head()

"""# Identification Aspect Category"""

print("Loading IndoBERT model...")
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")

"""## Seed Word"""

# Define seed words for categories
seed_words = {
        'Dosen': ['pelajaran', 'pembelajaran', 'materi', 'perkuliahan', 'dosen',
                  'guru', 'ajar', 'belajar', 'beliau', 'waktu', 'pengajar', 'bapak', 'ibu', 'konsultasi', 'pembimbing', 'metode'],
        'Administrasi': ['birokrasi', 'kantor', 'kepegawaian', 'humas', 'tu ', 'staff',
                         'informasi', 'administrasi', 'sponsor', 'surat', 'biku', 'biak',
                         'pembayaran', 'uang', 'kuliah', 'sks', 'bayar', 'krs', 'registrasi', 'keuangan'],
        'Fasilitas': ['fasilitas', 'lab', 'studio', 'perpustakaan', 'kelas', 'wifi',
                      'toilet', 'wc', 'kelas', 'kursi', 'meja', 'ekskalator',
                      'lift', 'cafetaria', 'foodcourt', 'parkir', 'kontak', 'ac', 'gdrive',
                      'poliklinik', 'akun', 'mhs', 'email', 'drive', 'software', 'hardware',
                      'komputer', 'ruangan', 'dinding', 'kantin', 'tangga', 'gedung', 'kulino',
                      'matkul', 'd ', 'h ', 'pintu', 'parkir', 'jaringan', 'bersih'],
        'Pembelajaran Offline': ['belajar di tempat', 'kelas', 'mengajar', 'belajar', 'jam', 'jadwal', 'ujian',
                                 'absen', 'ajar', 'offline', 'praktek', 'kurikulum', 'presentasi', 'absensi', 'materi'],
        'Pembelajaran Online': ['online', 'berkursus', 'mata kuliah', 'webinar', 'kulino', 'siadin',
                                'ujian online', 'gdrive', 'zoom', 'pembelajaran', 'form', 'website', 'google', 'virtual', 'meet', 'email', 'drive'],
        'Organisasi Mahasiswa': ['kampus', 'ormawa', 'organisasi', 'camp', 'kem', 'rkt', 'apbn', 'lpj', 'fik', 'sponsor', 'event', 'orma',
                                 'anggota', 'divisi', 'program kerja']
}

def get_sentence_embedding(sentence):
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    embeddings = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Precompute category embeddings
category_embeddings = {
    cat: np.mean([get_sentence_embedding(word) for word in words], axis=0)
    for cat, words in seed_words.items()
}

"""## Aspect Category"""

# Function to determine aspect category
def aspect_category(sentence_tokens):
    sentence = " ".join(sentence_tokens)  # Convert tokenized list to string
    sentence_embedding = get_sentence_embedding(sentence)

    similarities = {
        cat: cosine_similarity([sentence_embedding], [embed])[0][0]
        for cat, embed in category_embeddings.items()
    }
    best_category = max(similarities, key=similarities.get)
    return best_category if similarities[best_category] > 0.2 else None

"""## Save the category"""

# Apply the function to assign categories and store in a new column
data_fix['category'] = [aspect_category(opinion) for opinion in data_fix['opinion_tokenized']]

display(data_fix)

category_counts = data_fix['category'].value_counts()
category_counts

plt.figure(figsize = (7, 7))
plt.pie(data_fix['category'].value_counts(), labels = data_fix['category'].value_counts().index, autopct = '%1.1f%%', startangle=180, counterclock=False)
plt.legend(data_fix['category'].value_counts().index, loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Data Distribution by Category", fontsize = 15, fontweight= "bold")
plt.show()

plt.figure(figsize=(8, 6))
category_counts.plot(kind='bar')
plt.title('Number of Data per Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Path langsung ke lokasi file di Google Drive
file_path = '/content/drive/MyDrive/Aspect Category Sentiment Analysis/Dataset/dataset_category.csv'  # Ganti dengan path sesuai kebutuhan Anda

# Ekspor DataFrame ke CSV
data_fix.to_csv(file_path, index=False)

data_sent = data_fix

"""# Sentiment Polarity Labelling"""

import torch.nn.functional as F

# Muat model dan tokenizer IndoBERT
sentiment_tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-large-p1")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-large-p1")

# Muat INSET lexicon dari file .tsv
positive_lexicon = pd.read_csv('/content/drive/MyDrive/Aspect Category Sentiment Analysis/positive.tsv', sep='\t', header=None)
negative_lexicon = pd.read_csv('/content/drive/MyDrive/Aspect Category Sentiment Analysis/negative.tsv', sep='\t', header=None)

# Konversi ke set untuk pencarian cepat
positive_words = set(positive_lexicon[0].str.lower().tolist())
negative_words = set(negative_lexicon[0].str.lower().tolist())

# Fungsi untuk analisis sentimen dengan INSET dan pembobotan
def analyze_sentiment_with_inset(text):
    # Analisis sentimen menggunakan IndoBERT
    inputs = sentiment_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = sentiment_model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    sentiment_score = torch.argmax(probs, dim=-1).item()  # 0: Negatif, 1: Netral, 2: Positif

    # Hitung jumlah kata positif dan negatif menggunakan INSET
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    # Penyesuaian skor berdasarkan INSET
    if negative_count > positive_count:
        sentiment_score = max(0, sentiment_score - 1)  # Geser ke negatif jika lebih banyak kata negatif
    elif positive_count > negative_count:
        sentiment_score = min(2, sentiment_score + 1)  # Geser ke positif jika lebih banyak kata positif

    # Interpretasikan skor
    if sentiment_score == 0:
        return "Negative"
    elif sentiment_score == 1:
        return "Neutral"
    else:
        return "Positive"

# Terapkan analisis sentimen
data_sent['polarity'] = data_sent['opinion_stemmer'].apply(analyze_sentiment_with_inset)

# Tampilkan hasil
display(data_sent)

plt.figure(figsize = (7, 7))
plt.pie(data_sent['polarity'].value_counts(), labels = data_sent['polarity'].value_counts().index, autopct = '%1.1f%%', startangle=180, counterclock=False)
plt.legend(data_sent['polarity'].value_counts().index, loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Data Distribution by Polarity", fontsize = 15, fontweight= "bold")
plt.show()

polarity_counts = data_sent['polarity'].value_counts()
polarity_counts

plt.figure(figsize=(8, 6))
polarity_counts.plot(kind='bar')
plt.title('Number of Data per Polarity')
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""## Save dataset"""

# Path langsung ke lokasi file di Google Drive
file_path = '/content/drive/MyDrive/Aspect Category Sentiment Analysis/Dataset/dataset_polarity_before.csv'  # Ganti dengan path sesuai kebutuhan Anda

# Ekspor DataFrame ke CSV
data_sent.to_csv(file_path, index=False)

"""# Oversampling"""

data_model = data_sent.copy()
display(data_model)

label_encoder = LabelEncoder()
data_model['polarity_encoded'] = label_encoder.fit_transform(data_model['polarity'])

# Oversampling menggunakan RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_balanced, y_balanced = oversampler.fit_resample(
    data_model[['opinion_tokenized', 'opinion_stemmer', 'category']],  # Fitur teks
    data_model['polarity_encoded']  # Label kategori (encoded)
)

# Buat DataFrame dari hasil oversampling
data_model_balanced = pd.DataFrame(X_balanced, columns=['opinion_tokenized', 'opinion_stemmer', 'category'])
data_model_balanced['polarity_encoded'] = y_balanced

# Mapping kembali label encoded ke nama asli 'polarity'
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
data_model_balanced['polarity'] = data_model_balanced['polarity_encoded'].map(label_mapping)

# Simpan kembali dalam data_model
data_model = data_model_balanced
display(data_model)

# # Ambil mapping dari encoded ke kategori
# label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
# label_mapping

data_model_balanced['polarity'].value_counts().plot(kind='bar')
plt.title('Number of Data per Polarity (After Oversampling)')
plt.xlabel('Polarity')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

data_model = data_model.drop(['polarity_encoded'], axis=1)

display(data_model)

"""# Splitting Data"""

# Replace None values in 'category' with a string placeholder, like 'Unknown'
train_data, val_data = train_test_split(
    data_model.fillna({'polarity': 'Unknown'}),  # Replace None with 'Unknown'
    test_size=0.20,
    stratify=data_model['polarity'].fillna('Unknown'),  # Replace None with 'Unknown' for stratify
    random_state=42
)

display(train_data)

train_data.info()

val_data.info()

val_data, test_data = train_test_split(
    val_data,
    test_size=0.10,
    stratify=val_data['polarity'],
    random_state=42
)

val_data.info()

test_data.info()

# test_data = test_data.drop(columns=['opinion'])
display(test_data)

"""# Sentiment Polarity Analysis"""

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p2", num_labels=3)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['opinion_stemmer'], padding="max_length", truncation=True, max_length=128)

# Convert data to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

train_dataset = train_dataset.rename_column("polarity", "labels")
val_dataset = val_dataset.rename_column("polarity", "labels")

print(train_dataset["labels"][:5])  # Lihat beberapa nilai awal

label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}

train_dataset = train_dataset.map(lambda x: {"labels": label_mapping[x["labels"]]})
val_dataset = val_dataset.map(lambda x: {"labels": label_mapping[x["labels"]]})

print(train_dataset["labels"][:5])

print(type(train_dataset["labels"][0]))

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Format datasets for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

print(train_dataset.column_names)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# Define training arguments manually
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,  # Set manually
    per_device_train_batch_size=16,  # Set manually
    num_train_epochs=10,  # Set manually
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=30,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Ekstrak log pelatihan
logs = trainer.state.log_history
logs_df = pd.DataFrame(logs)

# Tampilkan log yang tersedia
logs_df

# Filter training loss dan validation loss dari log
train_loss = logs_df[logs_df["loss"].notnull()][["epoch", "loss"]]
val_loss = logs_df[logs_df["eval_loss"].notnull()][["epoch", "eval_loss"]]

# Plot training loss dan validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss["epoch"], train_loss["loss"], label="Training Loss", marker="o")
plt.plot(val_loss["epoch"], val_loss["eval_loss"], label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Extract validation accuracy from logs
val_accuracy = logs_df[logs_df["eval_accuracy"].notnull()][["epoch", "eval_accuracy"]]

# Plot validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(val_accuracy["epoch"], val_accuracy["eval_accuracy"], label="Validation Accuracy", marker="o", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.show()

val_results = trainer.evaluate(val_dataset)
print("Validation Results:", val_results)

val_results

# Predict on test dataset
test_predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(test_predictions.predictions, axis=1)

# Display predictions
print(predicted_labels)

# Convert predicted_labels to a list
predicted_labels = predicted_labels.tolist()

# Add the predicted_labels to the test_dataset using Dataset.map
test_dataset = test_dataset.map(lambda x, idx: {"predicted_labels": predicted_labels[idx]}, with_indices=True)

# Convert the test dataset to a DataFrame for easier viewing
test_df = test_dataset.to_pandas()

# Display the DataFrame
display(test_df)

display(test_df)

print(test_dataset.column_names)

test_dataset = Dataset.from_pandas(test_data)
label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
test_dataset = test_dataset.map(lambda x: {"labels": label_mapping[x["polarity"]]})

test_dataset = test_dataset.map(lambda x, idx: {"predicted_labels": predicted_labels[idx]}, with_indices=True)

test_df = test_dataset.to_pandas()
test_df.head()

cm = confusion_matrix(test_df["labels"], test_df["predicted_labels"], labels=[0, 1, 2])
cmd = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Neutral", "Positive"])

plt.figure(figsize=(8, 6))
cmd.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Classification Report
print("Classification Report:")
print(classification_report(test_df["labels"], test_df["predicted_labels"], target_names=["Negative", "Neutral", "Positive"]))

# Additional Metrics
accuracy = accuracy_score(test_df["labels"], test_df["predicted_labels"])
precision = precision_score(test_df["labels"], test_df["predicted_labels"], average='weighted')
recall = recall_score(test_df["labels"], test_df["predicted_labels"], average='weighted')
f1 = f1_score(test_df["labels"], test_df["predicted_labels"], average='weighted')

print("\nMetrics Summary:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

