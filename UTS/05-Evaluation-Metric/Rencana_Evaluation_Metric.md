# RENCANA EVALUATION METRIC

**Topik:** Klasifikasi tipe dan warna kartu *Digimon Card Game* — CNN dari awal vs *transfer learning* (MobileNetV2, ResNet-50, EfficientNet-B0)

## 1. Prinsip Evaluasi

1. **Dua tugas dinilai terpisah.** Tipe (4 kelas) dan warna (7 kelas) masing-masing punya metrik sendiri, ditambah satu metrik gabungan (*exact match*).
2. **Kelas kecil harus ikut dihitung.** Data sangat tidak seimbang (Digimon 73,6% kartu, putih 5,2%), sehingga metrik utama adalah *macro F1* [19], bukan *accuracy* saja.
3. **Tanpa kebocoran.** Gambar reguler dan *parallel* dari nomor kartu yang sama tidak pernah terpisah ke data latih dan data uji [18]. Data validasi hanya dipakai untuk *early stopping*, tidak pernah untuk pelaporan.
4. **Perbandingan adil.** Semua model memakai data, pembagian, augmentasi, dan *seed* yang identik.
5. **Perbedaan harus diuji.** Selisih antar-model dinyatakan "lebih baik" hanya jika lolos uji McNemar [20], [21] setelah koreksi Holm.
6. **Hasil negatif tetap dilaporkan.** Model yang tidak lebih baik, atau penurunan besar di skenario S2/S3, adalah temuan yang sah.

## 2. Notasi

| Simbol | Arti |
|---|---|
| $C$ | Jumlah kelas (4 untuk tipe, 7 untuk warna) |
| $N$ | Jumlah gambar uji |
| $TP_c, FP_c, FN_c$ | *True positive*, *false positive*, *false negative* untuk kelas $c$ |
| $n_c$ | Jumlah gambar uji yang kelas sebenarnya $c$ |
| $t_i, \hat t_i$ | Tipe sebenarnya dan tipe tebakan gambar ke-$i$ |
| $w_i, \hat w_i$ | Warna sebenarnya dan warna tebakan gambar ke-$i$ |
| $b, c'$ | Jumlah gambar yang benar di model A tetapi salah di model B, dan sebaliknya |

## 3. Metrik Klasifikasi per Tugas

| Metrik | Rumus | Makna |
|---|---|---|
| *Precision* kelas $c$ | $P_c = \dfrac{TP_c}{TP_c + FP_c}$ | Dari tebakan kelas $c$, berapa yang benar |
| *Recall* kelas $c$ | $R_c = \dfrac{TP_c}{TP_c + FN_c}$ | Dari kartu kelas $c$, berapa yang tertebak |
| F1 kelas $c$ | $F1_c = \dfrac{2 P_c R_c}{P_c + R_c}$ | Rata-rata harmonik *precision* dan *recall* |
| ***Macro F1*** (utama) | $\dfrac{1}{C}\sum_{c=1}^{C} F1_c$ | Setiap kelas berbobot sama |
| ***Accuracy*** (utama) | $\dfrac{1}{N}\sum_{c} TP_c$ | Proporsi tebakan benar |
| *Balanced accuracy* | $\dfrac{1}{C}\sum_{c=1}^{C} R_c$ | Rata-rata *recall* per kelas |
| *Weighted F1* | $\sum_{c=1}^{C} \dfrac{n_c}{N} F1_c$ | F1 berbobot jumlah data per kelas |
| *Confusion matrix* | Tabel label sebenarnya × tebakan | Pola kesalahan antar-kelas |

**Mengapa *macro F1*:** model yang selalu menebak "Digimon" sudah mendapat *accuracy* tipe sekitar 71% (per gambar), tetapi *macro F1*-nya hanya sekitar 0,21.

## 4. Metrik Gabungan dan Ketahanan

| Metrik | Rumus | Makna |
|---|---|---|
| *Exact match* | $\dfrac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat t_i = t_i \wedge \hat w_i = w_i]$ | Kartu terpilah benar hanya jika tipe **dan** warnanya benar |
| Penurunan pada ilustrasi *parallel* | $\Delta_{S2} = \overline{F1}_{S1} - \overline{F1}_{S2}$ | Positif = performa turun pada ilustrasi baru |
| Penurunan pada set baru | $\Delta_{S3} = \overline{F1}_{S1} - \overline{F1}_{S3}$ | Positif = performa turun pada set yang tidak dilatih |
| Selisih overlay | $\Delta_{WM} = F1_{\text{tanpa overlay}} - F1_{\text{dengan overlay}}$ | Indikasi pengaruh watermark (eksploratif) |
| Waktu inferensi | ms per gambar (CPU dan GPU) | Kelayakan dipakai pada sistem pemilahan |

## 5. Uji Signifikansi Statistik

**Uji McNemar** membandingkan dua model pada gambar uji yang sama (berpasangan), memakai prediksi gabungan seluruh *fold* sehingga setiap gambar diprediksi tepat satu kali [20], [21]:

$$
\chi^2 = \frac{(|b - c'| - 1)^2}{b + c'}
$$

Jika $b + c' < 25$, dipakai versi *exact* (binomial). Taraf signifikansi $\alpha = 0{,}05$. Karena ada 6 pasangan model untuk setiap tugas, nilai-*p* dikoreksi dengan **metode Holm** untuk mengendalikan kesalahan tipe I akibat banyak perbandingan.

| Hipotesis | Uji | Data |
|---|---|---|
| H1: *transfer learning* > CNN dari awal | McNemar + Holm | Prediksi gabungan S1 |
| H2: perbedaan antar-arsitektur *transfer learning* | McNemar + Holm | Prediksi gabungan S1 |
| H3: S2 < S1 | Selisih *macro F1* + IK 95% (*bootstrap* 1.000 kali, *resampling* per nomor kartu) | S1 vs S2 |
| H4: dengan overlay ≠ tanpa overlay | Selisih *macro F1* + IK 95% (*bootstrap*) | S1 per kelompok overlay |

Setiap metrik juga dilaporkan sebagai **rata-rata ± simpangan baku** dari 5 putaran.

## 6. Analisis Interpretabilitas (Grad-CAM)

Grad-CAM [17] diterapkan pada lapisan konvolusi terakhir model terbaik, untuk sampel prediksi benar dan salah dari setiap kelas.

| Kepala | Area yang diharapkan | Tanda jalan pintas |
|---|---|---|
| Warna | Bar nama kartu, terutama bagian kiri | Ilustrasi, tulisan "SAMPLE", watermark toko |
| Tipe | Tata letak: kotak Lv., angka DP, kotak *security*, tulisan tipe | Ilustrasi atau watermark |

## 7. Aturan Keputusan (Kriteria Keberhasilan)

| Pertanyaan | Kriteria |
|---|---|
| Apakah model "berhasil" mengklasifikasikan? | *Macro F1* S1 jauh di atas *baseline* tebak-kelas-terbanyak (≈ 0,21 untuk tipe) dan *recall* setiap kelas > 0 |
| Apakah *transfer learning* lebih baik? | H₀ hipotesis 1 ditolak (McNemar, *p* Holm < 0,05) |
| Apakah model tahan terhadap ilustrasi *parallel*? | IK 95% dari $\Delta_{S2}$ mencakup 0 atau $\Delta_{S2}$ kecil dibanding variasi antar-*fold* |
| Apakah model belajar dari desain kartu? | Grad-CAM menyala di bar nama dan tata letak pada sebagian besar sampel |
| Model mana yang direkomendasikan? | *Macro F1* tertinggi; bila tidak berbeda signifikan, pilih yang paling ringan (waktu inferensi terendah) |

## 8. Evaluasi Tahap Lanjutan (UAS, opsional)

| Evaluasi | Tujuan |
|---|---|
| Foto kamera kartu fisik (±100–200 foto) | Mengukur penurunan performa dari gambar katalog ke foto nyata |
| Set tambahan tanpa overlay "SAMPLE" | Memisahkan pengaruh watermark dari pengaruh set |
| Verifikasi label kedua oleh orang lain | Mengukur kesepakatan antar-pelabel |

## 9. Ancaman terhadap Validitas

| Ancaman | Penanganan |
|---|---|
| Kebocoran antar-varian | Pembagian per nomor kartu; pemeriksaan otomatis di setiap putaran |
| Label salah | Verifikasi ulang; jumlah koreksi dilaporkan |
| Watermark tercampur dengan set | Grad-CAM sebagai bukti pendukung; dinyatakan sebagai keterbatasan |
| Kelas sangat kecil (Digi-Egg putih hanya 1 kartu) | *Macro F1*, bobot kelas, stratifikasi gabungan; tidak menarik kesimpulan per kelas dari 1–2 sampel |
| Variasi hasil akibat *seed* | Rata-rata 5 putaran; *seed* dicatat |
| Penyetelan berlebihan pada data uji | *Hyperparameter* ditetapkan sebelum eksperimen; data uji tidak dipakai untuk memilih model |
| Gambar katalog ≠ foto nyata | Dinyatakan sebagai batasan; evaluasi foto kamera di tahap lanjutan |

## 10. Hasil Pilot (Uji Kelayakan Pipeline)

M1 (CNN dari awal) pada skenario S1 *fold* 0: 474 gambar latih, 83 gambar validasi, 140 gambar uji. Pelatihan berjalan 25 *epoch* (bobot terbaik di *epoch* 17) di CPU 2 inti tanpa GPU, total 18,6 menit. Konfigurasi sama dengan rencana, kecuali batas *epoch* diturunkan dari 60 menjadi 30 (sabar 8) karena keterbatasan waktu CPU. Model *transfer learning* (M2–M4) belum dijalankan karena bobot *pretrained* belum dapat diunduh di lingkungan penyusun; ketiganya dijalankan di Google Colab pada tahap berikutnya.

| Metrik (data uji *fold* 0, 140 gambar) | M1 CNN dari awal | *Baseline* tebak kelas terbanyak |
|---|---:|---:|
| *Accuracy* tipe | 0,764 | 0,721 |
| ***Macro F1* tipe** | **0,666** | 0,210 |
| *Balanced accuracy* tipe | 0,740 | 0,250 |
| *Accuracy* warna | 0,743 | 0,200 |
| ***Macro F1* warna** | **0,712** | 0,048 |
| *Balanced accuracy* warna | 0,740 | 0,143 |
| *Exact match* (tipe dan warna benar) | 0,529 | — |

*Baseline* tebak kelas terbanyak selalu menebak "Digimon" untuk tipe dan "merah" untuk warna (kelas terbanyak di data latih).

**F1 per kelas — tipe:**

| Kelas | Precision | Recall | F1 | Jumlah uji |
|---|---:|---:|---:|---:|
| Digimon | 0,910 | 0,802 | 0,853 | 101 |
| Tamer | 0,625 | 0,500 | 0,556 | 20 |
| Option | 0,833 | 0,909 | 0,870 | 11 |
| Digi-Egg | 0,261 | 0,750 | 0,387 | 8 |

**F1 per kelas — warna:**

| Kelas | Precision | Recall | F1 | Jumlah uji |
|---|---:|---:|---:|---:|
| merah | 0,950 | 0,679 | 0,792 | 28 |
| biru | 0,950 | 0,760 | 0,844 | 25 |
| kuning | 0,700 | 1,000 | 0,824 | 21 |
| hijau | 1,000 | 0,913 | 0,955 | 23 |
| hitam | 0,424 | 0,737 | 0,538 | 19 |
| ungu | 0,800 | 0,235 | 0,364 | 17 |
| putih | 0,545 | 0,857 | 0,667 | 7 |

**Interpretasi jujur:**

1. *Pipeline* berjalan dari awal sampai akhir: pembagian data per nomor kartu (tanpa kebocoran), augmentasi tanpa *flip* dan tanpa perubahan *hue*, pelatihan dua *head*, prediksi, dan perhitungan metrik.
2. Bahkan model paling sederhana yang dilatih dari awal dengan data sekecil ini sudah jauh di atas *baseline* tebak-kelas-terbanyak (*macro F1* tipe 0,666 vs 0,210; warna 0,712 vs 0,048). Artinya informasi tipe dan warna memang dapat dipelajari dari gambar.
3. Kelas tersulit pada pilot ini adalah tipe **Digi-Egg** (F1 0,387) dan warna **ungu** (F1 0,364). *Confusion matrix* menunjukkan pola yang jelas: 10 dari 17 kartu ungu ditebak hitam (keduanya gelap), dan 13 dari 101 kartu Digimon ditebak Digi-Egg (tata letak keduanya mirip, sama-sama memiliki kotak Lv.). Pola ini menjadi fokus analisis Grad-CAM dan perbandingan dengan model *transfer learning*.
4. *Exact match* gambar reguler 0,610 vs gambar *parallel* 0,325 pada *fold* ini. Angka ini belum setara skenario S2, karena pada S1 model juga melihat gambar *parallel* kartu lain saat latihan.
5. *Macro F1* gambar tanpa overlay "SAMPLE" (tipe 0,619, warna 0,624; 93 gambar) vs dengan overlay (tipe 0,782, warna 0,703; 47 gambar). Perbandingan ini masih tercampur dengan perbedaan set, sehingga belum dapat disimpulkan.
6. Ini **satu putaran** dengan **satu model**, sehingga belum ada kesimpulan tentang hipotesis. Hasil pilot hanya membuktikan kelayakan *pipeline* dan memberi titik acuan bagi M2–M4.

Berkas pendukung: `pilot_results.csv` (ringkasan), `pilot_per_kelas.csv` (metrik per kelas), `pilot_predictions_test.csv` (prediksi per gambar), `pilot_training_log.csv` (kurva pelatihan per *epoch*).

## 11. Alat Hitung

`scikit-learn` [25] (`f1_score`, `accuracy_score`, `balanced_accuracy_score`, `confusion_matrix`, `StratifiedGroupKFold`), `statsmodels` (`mcnemar`, `multipletests` untuk koreksi Holm), `pytorch-grad-cam` untuk Grad-CAM, dan `06-Kode/digimon_pipeline.py` (mode `ringkas`) untuk tabel hasil.

## Daftar Pustaka

[17] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 618–626, doi: 10.1109/ICCV.2017.74.

[18] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, p. 100804, 2023, doi: 10.1016/j.patter.2023.100804.

[19] H. He and E. A. Garcia, "Learning from imbalanced data," *IEEE Trans. Knowl. Data Eng.*, vol. 21, no. 9, pp. 1263–1284, 2009, doi: 10.1109/TKDE.2008.239.

[20] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947, doi: 10.1007/BF02295996.

[21] T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Comput.*, vol. 10, no. 7, pp. 1895–1923, 1998, doi: 10.1162/089976698300017197.

[25] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
