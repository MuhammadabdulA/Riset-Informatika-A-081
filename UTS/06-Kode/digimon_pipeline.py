"""
Pipeline klasifikasi tipe dan warna kartu Digimon Card Game dengan CNN.

Mode:
  uji_cepat  : enam uji kelayakan kode (data, kebocoran, loader, bentuk output, overfit, satu prediksi)
  pilot      : satu putaran (model + skenario + kunci) untuk uji kelayakan pipeline
  penuh      : 4 model x 3 skenario x 5 putaran = 60 pelatihan, hasil disimpan per putaran (bisa dilanjutkan)
  ringkas    : menghitung tabel metrik dari prediksi yang sudah tersimpan

Contoh:
  python digimon_pipeline.py --mode uji_cepat --data /content/data --meta dataset_metadata.csv --out hasil
  python digimon_pipeline.py --mode pilot --model cnn_sederhana --skenario S1 --kunci 0 --data ... --meta ... --out ...
  python digimon_pipeline.py --mode penuh --data ... --meta ... --out ...
  python digimon_pipeline.py --mode ringkas --out hasil
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

TIPE = ['Digimon', 'Tamer', 'Option', 'Digi-Egg']
WARNA = ['red', 'blue', 'yellow', 'green', 'black', 'purple', 'white']
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MODEL = ['cnn_sederhana', 'mobilenetv2_100', 'resnet50', 'efficientnet_b0']
SET = ['BT01', 'BT21', 'BT22', 'BT23', 'EX08']
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------- data
def muat_metadata(path):
    meta = pd.read_csv(path, keep_default_na=False)
    meta = meta[meta.duplicate_of == ''].reset_index(drop=True)          # buang duplikat
    meta['strata'] = meta.card_type + '_' + meta.color
    meta['fold'] = -1
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, idx) in enumerate(sgkf.split(meta, y=meta.strata, groups=meta.card_code)):
        meta.loc[idx, 'fold'] = fold
    assert meta.groupby('card_code').fold.nunique().max() == 1
    return meta


def bagi_latih_val(df_latih, seed=42):
    kartu = df_latih.drop_duplicates('card_code')
    k_latih, k_val = train_test_split(kartu.card_code, test_size=0.15, stratify=kartu.color, random_state=seed)
    return df_latih[df_latih.card_code.isin(k_latih)], df_latih[df_latih.card_code.isin(k_val)]


def data_skenario(meta, skenario, kunci):
    """S1 utama | S2 latih reguler, uji parallel | S3 leave-one-set-out. Hasil: (latih, val, uji)."""
    if skenario == 'S1':
        uji = meta[meta.fold == int(kunci)]
        latih, val = bagi_latih_val(meta[meta.fold != int(kunci)])
    elif skenario == 'S2':
        uji = meta[(meta.fold == int(kunci)) & (meta.is_parallel == 1)]
        latih, val = bagi_latih_val(meta[(meta.fold != int(kunci)) & (meta.is_parallel == 0)])
    elif skenario == 'S3':
        uji = meta[meta.folder_set == kunci]
        latih, val = bagi_latih_val(meta[meta.folder_set != kunci])
    else:
        raise ValueError(skenario)
    return latih, val, uji


def load_rgb(path):
    img = Image.open(path).convert('RGBA')                               # sudut kartu transparan
    putih = Image.new('RGBA', img.size, (255, 255, 255, 255))
    return Image.alpha_composite(putih, img).convert('RGB')


def jadikan_persegi(img, ukuran=256):
    w, h = img.size; s = max(w, h)
    kanvas = Image.new('RGB', (s, s), (255, 255, 255))
    kanvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return kanvas.resize((ukuran, ukuran), Image.LANCZOS)


CACHE = {}


def ambil_gambar(data_dir, image_id):
    if image_id not in CACHE:
        CACHE[image_id] = jadikan_persegi(load_rgb(os.path.join(data_dir, image_id)))
    return CACHE[image_id]


# Augmentasi: TANPA horizontal flip (warna pojok kiri) dan TANPA ubah saturation/hue (warna = label)
TF_LATIH = T.Compose([
    T.Resize((224, 224)),
    T.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
TF_UJI = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


class KartuDataset(Dataset):
    def __init__(self, df, tf, data_dir):
        self.df, self.tf, self.data_dir = df.reset_index(drop=True), tf, data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(ambil_gambar(self.data_dir, r.image_id)), TIPE.index(r.card_type), WARNA.index(r.color)


def loader(df, latih, data_dir, workers=2):
    return DataLoader(KartuDataset(df, TF_LATIH if latih else TF_UJI, data_dir),
                      batch_size=32, shuffle=latih, num_workers=workers)


# ----------------------------------------------------------------- model
class CNNSederhana(nn.Module):                     # M1: dilatih dari nol
    def __init__(self):
        super().__init__()

        def blok(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.fitur = nn.Sequential(blok(3, 32), blok(32, 64), blok(64, 128), blok(128, 256),
                                   nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3))
        self.head_tipe = nn.Linear(256, len(TIPE))
        self.head_warna = nn.Linear(256, len(WARNA))

    def forward(self, x):
        f = self.fitur(x)
        return self.head_tipe(f), self.head_warna(f)


class DuaKepala(nn.Module):                        # M2-M4: transfer learning, satu backbone + dua head
    def __init__(self, nama_backbone, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(nama_backbone, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.head_tipe = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, len(TIPE)))
        self.head_warna = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, len(WARNA)))

    def forward(self, x):
        f = self.backbone(x)
        return self.head_tipe(f), self.head_warna(f)


def buat_model(nama, pretrained=True):
    return CNNSederhana() if nama == 'cnn_sederhana' else DuaKepala(nama, pretrained)


# ----------------------------------------------------------------- latih
def bobot_kelas(label, n_kelas):
    hitung = np.bincount(label, minlength=n_kelas)
    return torch.tensor(len(label) / (n_kelas * np.maximum(hitung, 1)), dtype=torch.float32, device=DEV)


def jalankan_epoch(model, dl, lf_tipe, lf_warna, opt=None):
    latih = opt is not None
    model.train(latih)
    h = {'t_true': [], 't_pred': [], 'w_true': [], 'w_pred': []}
    total = 0.0
    with torch.set_grad_enabled(latih):
        for x, yt, yw in dl:
            x, yt, yw = x.to(DEV), yt.to(DEV), yw.to(DEV)
            out_t, out_w = model(x)
            loss = lf_tipe(out_t, yt) + lf_warna(out_w, yw)
            if latih:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(x)
            h['t_true'] += yt.tolist(); h['t_pred'] += out_t.argmax(1).tolist()
            h['w_true'] += yw.tolist(); h['w_pred'] += out_w.argmax(1).tolist()
    return total / len(dl.dataset), h


def skor_macro(h):
    return (f1_score(h['t_true'], h['t_pred'], average='macro') +
            f1_score(h['w_true'], h['w_pred'], average='macro')) / 2


def latih(model, dl_latih, dl_val, lf_tipe, lf_warna, epoch, lr, sabar=5, log=None):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    terbaik, sisa, simpan = -1.0, sabar, None
    for ep in range(epoch):
        loss_latih, _ = jalankan_epoch(model, dl_latih, lf_tipe, lf_warna, opt)
        loss_val, h_val = jalankan_epoch(model, dl_val, lf_tipe, lf_warna)
        skor = skor_macro(h_val)
        print(f'  epoch {ep + 1:2d} | loss latih {loss_latih:.3f} | loss val {loss_val:.3f} | macro F1 val {skor:.3f}', flush=True)
        if log is not None:
            log.append(dict(epoch=len(log) + 1, lr=lr, loss_latih=loss_latih, loss_val=loss_val, skor_val=skor))
        if skor > terbaik:
            terbaik, sisa = skor, sabar
            simpan = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            sisa -= 1
            if sisa == 0:
                print('  early stopping', flush=True); break
    model.load_state_dict(simpan)
    return model


def buat_dan_latih(nama, df_latih, df_val, data_dir, pretrained=True, epoch_maks=None, log=None):
    dl_latih, dl_val = loader(df_latih, True, data_dir), loader(df_val, False, data_dir)
    lf_tipe = nn.CrossEntropyLoss(weight=bobot_kelas([TIPE.index(t) for t in df_latih.card_type], len(TIPE)))
    lf_warna = nn.CrossEntropyLoss(weight=bobot_kelas([WARNA.index(w) for w in df_latih.color], len(WARNA)))
    model = buat_model(nama, pretrained).to(DEV)
    if nama == 'cnn_sederhana':                                         # satu tahap
        return latih(model, dl_latih, dl_val, lf_tipe, lf_warna, epoch=epoch_maks or 60, lr=1e-3, sabar=10, log=log)
    for p in model.backbone.parameters():                               # tahap 1: backbone dibekukan
        p.requires_grad = False
    latih(model, dl_latih, dl_val, lf_tipe, lf_warna, epoch=10, lr=1e-3, log=log)
    for p in model.backbone.parameters():                               # tahap 2: fine-tuning
        p.requires_grad = True
    return latih(model, dl_latih, dl_val, lf_tipe, lf_warna, epoch=epoch_maks or 30, lr=1e-4, log=log)


# ----------------------------------------------------------------- evaluasi
@torch.no_grad()
def prediksi(model, df, data_dir):
    model.eval()
    pt, pw = [], []
    for x, _, _ in loader(df, False, data_dir):
        out_t, out_w = model(x.to(DEV))
        pt += out_t.argmax(1).tolist(); pw += out_w.argmax(1).tolist()
    hasil = df[['image_id', 'card_code', 'folder_set', 'is_parallel', 'has_sample_overlay', 'card_type', 'color']].copy()
    hasil['pred_tipe'] = [TIPE[i] for i in pt]
    hasil['pred_warna'] = [WARNA[i] for i in pw]
    return hasil


def ringkas(df):
    return pd.Series({
        'acc_tipe': accuracy_score(df.card_type, df.pred_tipe),
        'f1_tipe': f1_score(df.card_type, df.pred_tipe, average='macro'),
        'bacc_tipe': balanced_accuracy_score(df.card_type, df.pred_tipe),
        'acc_warna': accuracy_score(df.color, df.pred_warna),
        'f1_warna': f1_score(df.color, df.pred_warna, average='macro'),
        'bacc_warna': balanced_accuracy_score(df.color, df.pred_warna),
        'exact_match': ((df.card_type == df.pred_tipe) & (df.color == df.pred_warna)).mean(),
    })


def uji_mcnemar(pred, model_a, model_b, kolom_benar, kolom_pred, skenario='S1'):
    from statsmodels.stats.contingency_tables import mcnemar
    p = pred[pred.skenario == skenario]
    a = p[p.model == model_a].sort_values('image_id')
    b = p[p.model == model_b].sort_values('image_id')
    assert (a.image_id.values == b.image_id.values).all()
    benar_a = (a[kolom_benar] == a[kolom_pred]).values
    benar_b = (b[kolom_benar] == b[kolom_pred]).values
    n_ab, n_ba = int((benar_a & ~benar_b).sum()), int((~benar_a & benar_b).sum())
    tabel = [[int((benar_a & benar_b).sum()), n_ab], [n_ba, int((~benar_a & ~benar_b).sum())]]
    return n_ab, n_ba, mcnemar(tabel, exact=(n_ab + n_ba) < 25).pvalue


# ----------------------------------------------------------------- mode
def mode_uji_cepat(meta, a):
    print('Uji 1 - data:', len(meta), 'gambar;', 'file hilang:',
          sum(not os.path.exists(os.path.join(a.data, p)) for p in meta.image_id))
    for skenario, daftar in [('S1', range(5)), ('S2', range(5)), ('S3', SET)]:
        total = 0
        for k in daftar:
            l, v, u = data_skenario(meta, skenario, k)
            kl, kv, ku = set(l.card_code), set(v.card_code), set(u.card_code)
            assert not (kl & ku) and not (kv & ku) and not (kl & kv), f'bocor di {skenario} {k}'
            total += len(u)
        print(f'Uji 2 - {skenario}: aman, total gambar uji = {total}')
    x, _, _ = next(iter(loader(meta.head(64), True, a.data, 0)))
    print('Uji 3 - batch:', tuple(x.shape))
    for nama in MODEL:
        m = buat_model(nama, a.pretrained).to(DEV).eval()
        with torch.no_grad():
            ot, ow = m(torch.randn(2, 3, 224, 224, device=DEV))
        assert ot.shape == (2, 4) and ow.shape == (2, 7)
        print(f'Uji 4 - {nama}: output {tuple(ot.shape)} {tuple(ow.shape)}')
    set_seed()
    kecil = meta[meta.is_parallel == 0].groupby('color').sample(3, random_state=42)
    dl = DataLoader(KartuDataset(kecil, TF_UJI, a.data), batch_size=32, shuffle=True)
    lf = nn.CrossEntropyLoss()
    m = buat_model('mobilenetv2_100', a.pretrained).to(DEV)
    latih(m, dl, dl, lf, lf, epoch=60, lr=1e-3, sabar=60)
    _, h = jalankan_epoch(m, dl, lf, lf)
    print('Uji 5 - overfit, macro F1 pada data sendiri:', round(skor_macro(h), 3), '(harus mendekati 1.0)')
    print('Uji 6 - jalankan --mode pilot untuk satu putaran sungguhan')


def jalankan_putaran(meta, a, nama, skenario, kunci):
    set_seed()
    df_latih, df_val, df_uji = data_skenario(meta, skenario, kunci)
    log = []
    t0 = time.time()
    model = buat_dan_latih(nama, df_latih, df_val, a.data, a.pretrained, a.epoch_maks, log)
    h = prediksi(model, df_uji, a.data)
    h['model'], h['skenario'], h['kunci'] = nama, skenario, str(kunci)
    return model, h, pd.DataFrame(log), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['uji_cepat', 'pilot', 'penuh', 'ringkas'], required=True)
    ap.add_argument('--data', help='folder berisi BT01, BT21, BT22, BT23, EX08')
    ap.add_argument('--meta', help='dataset_metadata.csv')
    ap.add_argument('--out', default='hasil')
    ap.add_argument('--model', default='cnn_sederhana', choices=MODEL)
    ap.add_argument('--skenario', default='S1', choices=['S1', 'S2', 'S3'])
    ap.add_argument('--kunci', default='0', help='nomor fold (S1, S2) atau nama set (S3)')
    ap.add_argument('--epoch_maks', type=int, default=None)
    ap.add_argument('--tanpa_pretrained', dest='pretrained', action='store_false')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    file_pred = os.path.join(a.out, 'prediksi_semua.csv')

    if a.mode == 'ringkas':
        pred = pd.read_csv(file_pred, keep_default_na=False)
        per = pred.groupby(['skenario', 'model', 'kunci']).apply(ringkas, include_groups=False)
        tabel = per.groupby(['skenario', 'model']).agg(['mean', 'std']).round(4)
        tabel.to_csv(os.path.join(a.out, 'tabel_hasil.csv'))
        print(tabel)
        return

    set_seed()
    meta = muat_metadata(a.meta)
    for p in meta.image_id:
        ambil_gambar(a.data, p)

    if a.mode == 'uji_cepat':
        mode_uji_cepat(meta, a)
    elif a.mode == 'pilot':
        model, h, log, detik = jalankan_putaran(meta, a, a.model, a.skenario, a.kunci)
        h.to_csv(os.path.join(a.out, 'pilot_predictions_test.csv'), index=False)
        log.to_csv(os.path.join(a.out, 'pilot_training_log.csv'), index=False)
        r = ringkas(h).round(4)
        r['durasi_menit'] = round(detik / 60, 1)
        pd.DataFrame([r]).to_csv(os.path.join(a.out, 'pilot_results.csv'), index=False)
        print(r)
    elif a.mode == 'penuh':
        selesai = set()
        if os.path.exists(file_pred):
            lama = pd.read_csv(file_pred, keep_default_na=False)
            selesai = set(zip(lama.model, lama.skenario, lama.kunci.astype(str)))
        putaran = [('S1', k) for k in range(5)] + [('S2', k) for k in range(5)] + [('S3', s) for s in SET]
        for nama in MODEL:
            for skenario, kunci in putaran:
                if (nama, skenario, str(kunci)) in selesai:
                    continue
                print(f'== {nama} | {skenario} | {kunci}', flush=True)
                model, h, _, _ = jalankan_putaran(meta, a, nama, skenario, kunci)
                h.to_csv(file_pred, mode='a', header=not os.path.exists(file_pred), index=False)
                if skenario == 'S1':
                    torch.save(model.state_dict(), os.path.join(a.out, f'bobot_{nama}_fold{kunci}.pt'))


if __name__ == '__main__':
    main()
