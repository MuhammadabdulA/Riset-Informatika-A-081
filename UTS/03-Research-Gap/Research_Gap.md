# RESEARCH GAP

**Topik:** Klasifikasi tipe dan warna kartu *Digimon Card Game* menggunakan *Convolutional Neural Network* (CNN)

> **Catatan metode.** Peta ini disusun dari penelusuran literatur terarah (Google Scholar, arXiv, portal jurnal SINTA, repositori universitas, dan sumber industri) dan dibaca pada tingkat abstrak/ringkasan, **bukan** *systematic review*. Tanda ✗ berarti "tidak disebut pada abstrak/ringkasan yang ditinjau", sehingga perlu diverifikasi pada naskah penuh sebelum dijadikan klaim final di paper.

---

## 1. Peta Literatur

### 1.1 Pengenalan kartu *trading card game* dan kartu permainan (paling dekat dengan topik)

| Ref | Studi | Objek | Pendekatan | Temuan | Keterbatasan terhadap topik ini |
|---|---|---|---|---|---|
| [1] | Zilio dkk. (2018), arXiv | Kartu *Magic: The Gathering* | CNN memprediksi warna dan tipe kartu dari ilustrasi dan teks | Atribut kartu dapat dipelajari dari gambar | Hanya MTG; tidak membahas varian ilustrasi, watermark, maupun kebocoran antar-varian |
| [2] | Nyckel (t.t.), layanan komersial | Kartu *Yu-Gi-Oh!* | Klasifikasi kategori kartu (Effect, Fusion, Link, dll.) dari foto | Model siap pakai | Metodologi, data, dan akurasi tidak dipublikasikan |
| [3] | Rosebrock (2021), PyImageSearch | Kartu *Yu-Gi-Oh!* (10.856 kartu) | *Triplet network* ResNet-101 + *re-ranking* ORB | Mengenali lebih dari 10.000 kartu; mayoritas kartu hanya punya satu ilustrasi | Identifikasi kartu, bukan atribut; non-akademik tanpa protokol evaluasi formal |
| [4] | Pua (2024), laporan proyek Stanford CS231n | Kartu *Pokémon* dari rekaman turnamen | *Fine-tuning* YOLOv8 segmentasi dengan data sintetis | mAP50-95 0,441; kartu foil (*holographic*) sering terlewat | Deteksi/segmentasi 10 kartu, bukan klasifikasi atribut |
| [5] | Wzorek & Kryjak (2021), arXiv | Kartu remi (bridge) | Dataset sintetis + deteksi YOLOv4 | Efektivitas deteksi 99,8% | 52 kelas tetap; tanpa atribut warna/tipe |
| [6] | Čampelj (2022), skripsi Univ. Ljubljana | Kartu Tarok | YOLO + dataset sintetis, divalidasi dengan gambar nyata | Penghitung poin otomatis | Kelas sedikit dan tetap |

### 1.2 Penelitian lokal (SINTA) klasifikasi citra dengan CNN dan *transfer learning*

| Ref | Studi | Objek | Model | Temuan | Keterbatasan terhadap topik ini |
|---|---|---|---|---|---|
| [7] | Herlawati dkk. (2026), *J. Komtika* | Daun herbal dan *undertone* kulit wajah | ResNet50 vs MobileNetV2 | ResNet50 95,00% vs MobileNetV2 94,09% (daun); 89,3% vs 68,3% (kulit) | Selisih antar-arsitektur sangat bergantung domain; tanpa uji signifikansi |
| [8] | Rayhan dkk. (2024), *J. Cerita* | Makanan tradisional Indonesia | EfficientNetV2 vs MobileNetV2 | EfficientNetV2 31% vs MobileNetV2 99% | Selisih ekstrem tanpa analisis penyebab; tanpa *cross-validation* |
| [9] | Hartono & Muslikh (2025), *JISAD* | 20 jenis buah (4.000 citra) | MobileNetV2 *transfer learning* | Akurasi ±94%; precision, recall, F1 rata-rata 0,93–0,94 | Satu arsitektur; kelas seimbang |
| [10] | Supriyanti & Anggoro (2021), *Khazanah Informatika* | Tokoh Pandawa pada citra wayang (430 citra, 4 kelas) | CNN dari nol (3 lapis konvolusi) | Akurasi 0,93 | Dataset kecil tanpa *transfer learning*; satu kali pembagian data |

### 1.3 Landasan metodologis

| Ref | Studi | Peran dalam penelitian ini |
|---|---|---|
| [11], [12] | LeCun dkk. (2015); Deng dkk. (2009) | Dasar CNN dan dataset ImageNet untuk bobot *pretrained* |
| [13] | Yosinski dkk. (2014) | Fitur lapisan awal CNN dapat dipindahkan ke tugas lain (*transfer learning*) |
| [14]–[16] | He dkk. (2016); Sandler dkk. (2018); Tan & Le (2019) | Arsitektur ResNet-50, MobileNetV2, EfficientNet-B0 |
| [17] | Selvaraju dkk. (2017) | Grad-CAM untuk interpretabilitas |
| [18] | Kapoor & Narayanan (2023) | Kebocoran data sebagai sumber hasil yang terlalu optimistis |
| [19] | He & Garcia (2009) | Pembelajaran pada data tidak seimbang dan metrik yang sesuai |
| [20], [21] | McNemar (1947); Dietterich (1998) | Uji McNemar untuk membandingkan dua pengklasifikasi |

---

## 2. Matriks Celah (Gap Matrix)

Legenda: ✓ ada | ✗ tidak disebut pada ringkasan yang ditinjau | — tidak berlaku. **Kartu TCG:** objek berupa kartu *trading card game*; **Atribut kartu:** mengklasifikasikan atribut (tipe/warna/kategori), bukan identitas kartu; **≥ 2 arsitektur:** membandingkan minimal dua arsitektur CNN; **Varian ilustrasi:** menguji kartu bernomor sama dengan ilustrasi berbeda; **Split per grup:** pembagian data per nomor kartu/objek untuk mencegah kebocoran; **Metrik per kelas:** melaporkan *precision*/*recall*/F1 per kelas atau *macro F1*; **Uji statistik:** uji signifikansi antar-model.

| Studi | Kartu TCG | Atribut kartu | *Transfer learning* | ≥ 2 arsitektur | Varian ilustrasi | Split per grup | Metrik per kelas | Uji statistik | Grad-CAM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Zilio dkk. [1] | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Nyckel [2] | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Rosebrock [3] | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Pua [4] | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Wzorek & Kryjak [5] | ✗ | ✗ | ✗ | ✗ | — | ✗ | ✗ | ✗ | ✗ |
| Čampelj [6] | ✗ | ✗ | ✗ | ✗ | — | ✗ | ✗ | ✗ | ✗ |
| Herlawati dkk. [7] | ✗ | — | ✓ | ✓ | — | ✗ | ✗ | ✗ | ✗ |
| Rayhan dkk. [8] | ✗ | — | ✓ | ✓ | — | ✗ | ✗ | ✗ | ✗ |
| Hartono & Muslikh [9] | ✗ | — | ✓ | ✗ | — | ✗ | ✓ | ✗ | ✗ |
| Supriyanti & Anggoro [10] | ✗ | — | ✗ | ✗ | — | ✗ | ✗ | ✗ | ✗ |
| **Penelitian ini** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

---

## 3. Pernyataan Research Gap

Dari literatur yang berhasil ditinjau:

| Kode | Celah | Bukti |
|---|---|---|
| **G1 — Domain** | Belum ditemukan penelitian ilmiah klasifikasi kartu ***Digimon Card Game***, terutama versi bahasa Jepang. Penelitian TCG yang ada berfokus pada *Magic: The Gathering*, *Yu-Gi-Oh!*, dan *Pokémon*, dan sebagian besar non-akademik. Penelitian lokal (SINTA) yang ditinjau belum menyentuh domain kartu. | Bagian 1.1 dan 1.2 |
| **G2 — Varian ilustrasi** | Ketahanan model terhadap **ilustrasi *parallel*** (nomor kartu sama, ilustrasi berbeda) belum dievaluasi. Belum diketahui apakah model mengenali atribut dari desain kartu atau menghafal ilustrasinya. | Matriks celah, kolom "Varian ilustrasi" |
| **G3 — Protokol evaluasi** | Belum ditemukan studi kartu yang memakai **pembagian data berbasis grup** untuk mencegah kebocoran antar-varian, disertai **macro F1** untuk kelas tidak seimbang dan **uji signifikansi** antar-model. Studi lokal [7], [8] menunjukkan selisih antar-arsitektur yang sangat berbeda antar-domain tanpa uji statistik. | Matriks celah; [7], [8], [18] |
| **G4 — Data riil berwatermark** | Pengaruh elemen non-kartu (watermark toko dan overlay "SAMPLE") pada keputusan CNN belum dianalisis. Penelitian kartu sebelumnya memakai gambar resmi atau data sintetis. | Bagian 1.1; kolom "Grad-CAM" |
| **G5 — Aturan label kartu dua warna** | Belum ada aturan label kartu dua warna beserta dampaknya pada desain augmentasi: *horizontal flip* dan perubahan *hue* yang umum dipakai justru membuat label salah. | Bagian 1.1 dan 1.2 |

## 4. Posisi dan Kontribusi Penelitian

1. **Empiris:** studi klasifikasi tipe dan warna pertama untuk *Digimon Card Game* (dalam literatur yang ditinjau), membandingkan CNN dari awal dengan tiga model *transfer learning*.
2. **Metodologis:** protokol evaluasi bebas kebocoran (*stratified group 5-fold* per nomor kartu), skenario ketahanan terhadap ilustrasi *parallel* dan set baru, macro F1, dan uji McNemar.
3. **Interpretabilitas:** analisis Grad-CAM untuk memeriksa apakah model melihat bar nama dan tata letak kartu, bukan ilustrasi atau watermark.
4. **Sumber daya:** dataset 697 gambar berlabel tipe dan warna, beserta skrip pembangun metadata dan *pipeline* yang dapat direproduksi.
5. **Lokal:** menambah rujukan berbahasa Indonesia untuk klasifikasi citra pada domain hobi/koleksi.

## 5. Pertanyaan Riset ↔ Celah

| Rumusan masalah | Celah yang dijawab |
|---|---|
| RM1: performa CNN untuk tipe dan warna | G1, G3 |
| RM2: CNN dari awal vs *transfer learning*, signifikansi | G3 |
| RM3: ketahanan pada ilustrasi *parallel* dan set baru | G2, G5 |
| RM4: area perhatian model dan pengaruh watermark | G4 |

## 6. Dasar Metodologis Penelitian

Penelitian ini memakai CNN [11] dengan bobot *pretrained* ImageNet [12] berdasarkan temuan bahwa fitur visual dapat dipindahkan antar-tugas [13]. Arsitektur yang dibandingkan adalah ResNet-50 [14], MobileNetV2 [15], dan EfficientNet-B0 [16], ditambah CNN sederhana yang dilatih dari awal sebagai pembanding. Pembagian data dirancang untuk mencegah kebocoran [18], metrik dipilih untuk data tidak seimbang [19], perbedaan antar-model diuji dengan uji McNemar [20], [21], dan keputusan model dianalisis dengan Grad-CAM [17]. Studi kartu [1]–[6] dan studi lokal [7]–[10] menjadi acuan pembanding.

## 7. Catatan untuk Pengembangan Menjadi Paper (UAS)

- Lakukan penelusuran sistematis (kata kunci: *trading card recognition*, *card game image classification*, *Magic the Gathering CNN*, *Pokémon card detection*) di Scopus/IEEE Xplore untuk memperkuat klaim G1.
- Baca naskah penuh [1] untuk memastikan klaim ✗ pada matriks, khususnya apakah memakai *transfer learning* dan bagaimana data dibagi.
- Tambahkan studi lokal tentang klasifikasi warna objek dan penanganan data tidak seimbang sebagai pembanding metodologis.
- Jika memungkinkan, tambahkan data dari set tanpa overlay "SAMPLE" untuk memperkuat analisis G4.

## Daftar Pustaka

[1] F. Zilio, M. Prates, and L. Lamb, "Neural networks models for analyzing Magic: The Gathering cards," *arXiv preprint* arXiv:1810.03744, 2018.

[2] Nyckel, "Yugioh card category — pretrained classifier." [Online]. Available: https://edge.nyckel.com/pretrained-classifiers/yugioh-card-category/ (accessed Oct. 2026).

[3] A. Rosebrock, "An interview with Anthony Lowhur — Recognizing 10,000 Yugioh cards with computer vision and deep learning," *PyImageSearch*, Mar. 3, 2021. [Online]. Available: https://pyimagesearch.com/2021/03/03/an-interview-with-anthony-lowhur-recognizing-10000-yugioh-cards-with-computer-vision-and-deep-learning/

[4] E. Pua, "Real-time Pokémon card detection from tournament footage," Stanford CS231n course project report, 2024. [Online]. Available: https://cs231n.stanford.edu/2024/papers/real-time-pokemon-card-detection-from-tournament-footage.pdf

[5] P. Wzorek and T. Kryjak, "Training dataset generation for bridge game registration," *arXiv preprint* arXiv:2109.11861, 2021.

[6] M. Čampelj, "Sistem za razpoznavo igralnih kart na osnovi globokega učenja [Deep learning-based playing card recognition system]," Diplomsko delo, Fakulteta za strojništvo, Univerza v Ljubljani, Ljubljana, 2022.

[7] H. Herlawati, A. A. Hendharsetiawan, W. Priatna, A. P. Dzulqiyana, and R. A. Rahmadanti, "Komparasi kinerja ResNet50 dan MobileNetV2 pada klasifikasi citra multi-domain," *J. Komtika (Komputasi dan Informatika)*, vol. 10, no. 1, pp. 46–61, 2026, doi: 10.31603/komtika.v10.i1.16926.

[8] A. Rayhan, R. Hidayat, and R. Afyenni, "Perbandingan akurasi EfficientNetV2 dan MobileNetV2 pada klasifikasi makanan tradisional Indonesia," *J. Cerita*, vol. 10, no. 2, pp. 124–127, 2024, doi: 10.33050/cerita.v10i2.3326.

[9] A. C. Hartono and A. R. Muslikh, "Penerapan transfer learning MobileNetV2 pada klasifikasi citra jenis buah-buahan," *J. Inf. Syst. Appl. Dev. (JISAD)*, vol. 3, no. 2, pp. 103–111, 2025, doi: 10.26905/jisad.v3i2.16187.

[10] W. Supriyanti and D. A. Anggoro, "Classification of Pandavas figure in shadow puppet images using convolutional neural networks," *Khazanah Informatika*, vol. 7, no. 1, 2021, doi: 10.23917/khif.v7i1.12484.

[11] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436–444, 2015, doi: 10.1038/nature14539.

[12] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2009, pp. 248–255, doi: 10.1109/CVPR.2009.5206848.

[13] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?" in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 27, 2014, pp. 3320–3328.

[14] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[15] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474.

[16] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 97, 2019, pp. 6105–6114.

[17] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.

[18] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, p. 100804, 2023, doi: 10.1016/j.patter.2023.100804.

[19] H. He and E. A. Garcia, "Learning from imbalanced data," *IEEE Trans. Knowl. Data Eng.*, vol. 21, no. 9, pp. 1263–1284, 2009, doi: 10.1109/TKDE.2008.239.

[20] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947, doi: 10.1007/BF02295996.

[21] T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Comput.*, vol. 10, no. 7, pp. 1895–1923, 1998, doi: 10.1162/089976698300017197.
