"""
Membangun metadata dataset gambar kartu Digimon Card Game dari nama file.

Pemakaian:
    python build_metadata.py <folder_dataset> <output_csv> [label_tipe_warna.csv]

Jika file label diberikan, kolom card_type dan color diisi berdasarkan card_code
(kartu parallel otomatis mewarisi label kartu reguler-nya).

<folder_dataset> berisi subfolder per set (BT01, BT21, BT22, BT23, EX08),
masing-masing berisi gambar PNG dengan pola nama:
    <SET>-<NOMOR>.png          -> kartu reguler (base), contoh BT1-091.png
    <SET>-<NOMOR>_P<k>.png     -> kartu parallel ke-k, contoh BT1-089_P1.png
    P-<NOMOR>.png              -> kartu promo
"""
import csv
import hashlib
import re
import sys
from pathlib import Path

from PIL import Image

POLA = re.compile(r"^(?P<set>[A-Z]+\d*)-(?P<no>\d+)(?:_P(?P<par>\d+))?(?P<dup>\(\d+\))?\.png$")


def norm_set(kode: str) -> str:
    """BT1 -> BT01, EX8 -> EX08, P -> P (promo)."""
    m = re.match(r"^([A-Z]+)(\d*)$", kode)
    huruf, angka = m.groups()
    return f"{huruf}{int(angka):02d}" if angka else huruf


def baca_label(label_csv: Path) -> dict:
    with label_csv.open(encoding="utf-8") as f:
        return {r["card_code"]: r for r in csv.DictReader(f)}


def main(root: Path, out_csv: Path, label_csv: Path | None = None) -> None:
    label = baca_label(label_csv) if label_csv else {}
    baris, hash_terlihat = [], {}
    # File berakhiran "(1)" dll. (hasil unduh ganda) diurutkan terakhir agar
    # yang ditandai sebagai duplikat adalah salinannya, bukan file aslinya.
    semua = sorted(root.glob("*/*.png"), key=lambda p: ("(" in p.name, str(p)))
    for path in semua:
        m = POLA.match(path.name)
        if not m:
            print(f"[lewati] nama tidak sesuai pola: {path}")
            continue
        set_kartu = norm_set(m["set"])
        kode = f"{m['set']}-{m['no']}"
        par = int(m["par"]) if m["par"] else 0
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        with Image.open(path) as im:
            w, h = im.size
        duplikat_dari = hash_terlihat.get(md5, "")
        hash_terlihat.setdefault(md5, f"{path.parent.name}/{path.name}")
        baris.append({
            "image_id": f"{path.parent.name}/{path.name}",
            "folder_set": path.parent.name,
            "card_code": kode,
            "card_set": set_kartu,
            "card_no": m["no"],
            "variant": "base" if par == 0 else f"parallel_{par}",
            "is_parallel": int(par > 0),
            "is_reprint_or_promo": int(set_kartu != path.parent.name),
            "duplicate_of": duplikat_dari,
            "width": w,
            "height": h,
            "md5": md5,
            # Overlay "SAMPLE" ditemukan pada semua gambar set BT22 dan BT23 (tinjauan visual)
            "has_sample_overlay": int(path.parent.name in {"BT22", "BT23"}),
            "card_type": "",
            "color": "",
        })
    for b in baris:
        if b["card_code"] in label:
            b["card_type"] = label[b["card_code"]]["card_type"]
            b["color"] = label[b["card_code"]]["color"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
        w.writeheader()
        w.writerows(baris)
    print(f"{len(baris)} baris ditulis ke {out_csv}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]),
         Path(sys.argv[3]) if len(sys.argv) > 3 else None)
