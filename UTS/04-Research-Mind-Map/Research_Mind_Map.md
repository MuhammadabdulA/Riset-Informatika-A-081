# RESEARCH MIND MAP

**Topik:** Klasifikasi Tipe dan Warna Kartu *Digimon Card Game* — CNN dari awal vs *Transfer Learning*

## Versi gambar

![Research Mind Map](research_mind_map.png)

Versi pohon yang lebih rinci: [research_mind_map_detail.png](research_mind_map_detail.png).

## Versi teks (Mermaid, tampil otomatis di GitHub)

```mermaid
mindmap
  root((Klasifikasi Kartu Digimon))
    1. Latar Belakang dan Masalah
      Pemilahan kartu di toko TCG masih manual
      Riset kartu TCG fokus MTG, Yu-Gi-Oh, Pokemon
      Ketahanan terhadap kartu parallel belum diuji
      Kartu dua warna dan watermark toko
    2. Dataset
      697 gambar unik dari 503 kartu
      5 set - BT01, BT21, BT22, BT23, EX08
      503 reguler dan 194 parallel
      Label 4 tipe dan 7 warna pojok kiri
      Kelas tidak seimbang
    3. Preprocessing
      Deduplikasi MD5
      Padding persegi 224x224
      Tanpa horizontal flip dan ubah hue
      Stratified group 5-fold per nomor kartu
    4. Model
      M1 CNN dari awal
      M2 MobileNetV2
      M3 ResNet-50
      M4 EfficientNet-B0
      Satu backbone dua head
    5. Evaluasi
      Macro F1 dan accuracy
      Balanced accuracy dan confusion matrix
      Exact match tipe dan warna
      Uji McNemar dengan koreksi Holm
      Grad-CAM
    6. Skenario dan Kontribusi
      S1 utama 5-fold
      S2 latih reguler uji parallel
      S3 leave-one-set-out
      Kontribusi - dataset, protokol bebas kebocoran, analisis ketahanan
```

## Alur logika penelitian

```mermaid
flowchart TD
    G["Celah G1-G5"] --> RQ["Rumusan Masalah RM1-RM4"]
    RQ --> H["Hipotesis H1-H4"]
    H --> M["4 model x 3 skenario x 5 putaran"]
    M --> E["Evaluasi macro F1, accuracy, McNemar, Grad-CAM"]
    E --> K["Kesimpulan H0 ditolak atau gagal ditolak"]
```
