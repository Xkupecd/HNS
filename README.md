# HNS – Projekt 2 (skupina 9): Rozpoznávanie ľudských činností z videa

Rozpoznávanie 4 činností z videa, ktoré sa dajú názorne predviesť pred kamerou: **clap** (tlieskanie), **drink** (pitie), **eat** (jedenie), **wave** (mávanie). Dataset je [HMDB51](https://huggingface.co/datasets/Serrelab/hmdb51) (oficiálne vydanie od Serre Lab, pôvodný odkaz na brown.edu už nefunguje).

Porovnávame tri metódy, všetky nad rovnakou zamrznutou CNN (ConvNeXt-B, ImageNet-22k):

| | metóda | časový model |
|---|---|---|
| **M1** | CNN + SVM | priemer a smerodajná odchýlka príznakov 16 snímok → RBF SVM |
| **M2** | CNN + LSTM | jednovrstvové LSTM (256) nad postupnosťou príznakov snímok |
| **M3** | CNN + Transformer (hybrid) | 2 vrstvy self-attention nad snímkami, token [CLS] |

## Spustenie

Používa sa venv z Projektu 1 (`../Projekt1/venv`, torch 2.5.1 + CUDA). Do neho bol namiesto `opencv-python-headless` nainštalovaný `opencv-python`, lebo headless verzia nevie otvoriť okno kamery. Zoznam balíkov je v `requirements.txt`.

```bash
cd Projekt2
P=../Projekt1/venv/bin/python

# 1. dáta (už pripravené): 32 snímok 224x224 z každého videa + split 70/15/15
$P -m src.data_preprocessing.extract_frames
$P -m src.data_preprocessing.split_dataset

# 2. príznaky zamrznutej CNN (train: 10 augmentovaných verzií každého videa), ~3.5 min
$P -m src.features

# 3. trénovanie, vždy 5 behov
$P -m src.train_m1_cnn_svm
$P -m src.train_m2_cnn_lstm
$P -m src.train_m3_cnn_transformer

# 4. porovnanie a GradCAM
$P -m src.compare_models
$P -m src.gradcam

# 5. kamera
$P -m src.camera.demo                                  # živé rozpoznávanie (predvolene najlepší model)
$P -m src.camera.record_clips --person dominik --n 10  # nahratie vlastných klipov
$P -m src.camera.finetune_camera                       # test na kamere + dotrénovanie
$P -m src.camera.demo --model m1 --checkpoint Results/camera_finetune/M1_CNN_SVM/finetuned_svm.joblib
```

## Dáta a rozdelenie

| trieda | videá | train | val | test |
|---|---|---|---|---|
| clap | 130 | 92 | 21 | 17 |
| drink | 164 | 114 | 27 | 23 |
| eat | 108 | 80 | 14 | 14 |
| wave | 104 | 75 | 13 | 16 |
| **spolu** | **506** | **361 (71 %)** | **75 (15 %)** | **70 (14 %)** |

**Rozdelenie podľa zdrojového videa.** HMDB51 často strihá viac klipov z jedného filmu (napr. `The_Slow_Clap` má 31 klipov) s tou istou osobou, scénou a svetlom. Pri bežnom náhodnom splite by sa takmer rovnaké klipy dostali do train aj test a accuracy by vyšla nadhodnotená. Preto sú všetky klipy jedného zdroja (157 zdrojov) vždy v tej istej časti. Keďže skupiny sú veľmi nerovnaké, zo 20 000 náhodných priradení skupín sa vyberie to, ktoré je v každej triede najbližšie k 70/15/15 (`split_dataset.py`).

**Vstup modelu:** 16 snímok rovnomerne rozložených cez celý klip (HMDB klipy trvajú 1,5–4 s). Snímky sa zmenšia na 224×224 bez orezania, lebo videá majú šírku 320–560 px a stredový výrez by mohol odrezať ruky. Pri trénovaní sa používa časový jitter (náhodná snímka z každého zo 16 úsekov) a augmentácia, ktorá je rovnaká pre všetky snímky klipu (výrez, preklopenie, farby), aby sa nepokazil pohyb.

## Výsledky (test, 5 behov)

| model | Accuracy | Precision | Recall | F1 | Acc min | Acc max |
|---|---|---|---|---|---|---|
| **M1 CNN+SVM** | **82,57 ± 1,07** | **84,39 ± 0,47** | **82,60 ± 0,89** | **82,21 ± 1,03** | 81,43 | 84,29 |
| M2 CNN+LSTM | 76,29 ± 2,94 | 77,97 ± 3,18 | 75,62 ± 3,22 | 75,16 ± 3,32 | 72,86 | 80,00 |
| M3 CNN+Transformer | 81,43 ± 2,86 | 82,04 ± 3,23 | 81,02 ± 3,43 | 80,49 ± 3,23 | 80,00 | **87,14** |

Senzitivita a špecificita pre každú triedu (jedna trieda proti ostatným, priemer z 5 behov):

| trieda | M1 sens / spec | M2 sens / spec | M3 sens / spec |
|---|---|---|---|
| clap | 98,8 / 97,0 | 95,3 / 89,8 | 92,9 / 91,7 |
| drink | 80,9 / 91,5 | 79,1 / 87,7 | 84,3 / 92,3 |
| eat | 85,7 / 88,6 | 74,3 / 91,8 | 84,3 / 92,9 |
| wave | 65,0 / 99,6 | 53,8 / 98,5 | 62,5 / 98,2 |

Grafy a kontingenčné tabuľky: `Results/M1_to_M3/` (porovnanie), `Results/M*/best_run_confusion_matrix.png`, `Results/M*/run*/` (každý beh zvlášť).

**Pozorovania:**
- Test má iba 70 videí, takže 1 video znamená 1,4 p. b. Rozdiel medzi M1 a M3 (1,1 p. b.) je v rámci šumu. M3 má najlepší jednotlivý beh (87,1 %), M1 je najstabilnejší (smerodajná odchýlka 1,1).
- **M1 bez informácie o poradí snímok je najlepší v priemere.** Tieto 4 činnosti sa dajú rozlíšiť už podľa vzhľadu (pohár, jedlo, zdvihnutá ruka, ruky pred hrudníkom). S 361 trénovacími videami sa LSTM a Transformer nestihnú naučiť časové vzťahy, ktoré by priniesli viac, ako ich stojí preučenie.
- **LSTM (M2) je najslabší.** Pri oboch hlavách (M2 aj M3) je validačná loss najnižšia už v 1.–3. epoche a potom rastie (napr. M2 z 0,78 na 1,5–2,1), aj keď validačná accuracy sa ďalej mierne mení. Checkpoint podľa validačnej accuracy preto vyberá skoré epochy. Viac dát by pomohlo najviac práve časovým modelom.
- **Najťažšia je trieda wave** (senzitivita 54–65 %). Mávanie v HMDB často trvá iba zlomok klipu a ruka je malá alebo čiastočne mimo obrazu. Model ho zamieňa s clap. Drink a eat sa zamieňajú navzájom (ruka pri ústach).

## Výber backbonu a hyperparametrov

Všetko sa vyberalo iba podľa **validačných** dát, testovacie dáta sa použili až na konci.

Zamrznuté príznaky + SVM (rovnaký postup ako M1, validačná accuracy):

| backbone | predtrénovanie | val acc |
|---|---|---|
| ConvNeXt-T | ImageNet-1k | 0,613 |
| ConvNeXt-T | ImageNet-22k | 0,653 |
| ConvNeXt-T | DINOv3 | 0,627 |
| ConvNeXt-S | DINOv3 | 0,680 |
| **ConvNeXt-B** | **ImageNet-22k** | **0,733** |
| ConvNeXt-B | CLIP LAION-2B | 0,653 |
| ConvNeXt-B | DINOv3 | 0,627 |
| ConvNeXt-B | IN-22k → IN-1k | 0,693 |
| ConvNeXt-L | ImageNet-22k | 0,667 |

ImageNet-22k obsahuje aj triedy blízke činnostiam a predmety (poháre, jedlo), preto jeho príznaky fungujú lepšie ako čisté IN-1k. Väčší ConvNeXt-L už nepomohol a je 3× pomalší.

Hlavy M2/M3: scan lr ∈ {1e-4, 3e-4, 1e-3} × weight decay ∈ {0,01; 0,1} × LayerNorm na vstupe (áno/nie), priemer validačnej accuracy z 5 behov. Rozdiely boli malé (LSTM ~0,70 všade, Transformer 0,72–0,77). Pre obe hlavy sa použili rovnaké parametre: **AdamW, lr 1e-3, weight decay 0,1, kosínusový pokles, 30 epoch, batch 32, LayerNorm na vstupe**. Ukladá sa epocha s najlepšou validačnou accuracy (pri zhode s nižšou val loss).

### Ablácia: prvá verzia s dotrénovaním CNN

Prvá verzia používala ConvNeXt-T (ImageNet-1k) a pri M2/M3 dotrénovala aj posledný stupeň CNN (15,5 M parametrov, lr 3e-5, batch 8 videí, 20 epoch):

| model | accuracy (v1) | accuracy (finálna) |
|---|---|---|
| M1 CNN+SVM | 58,57 ± 1,28 | **82,57 ± 1,07** |
| M2 CNN+LSTM | 62,29 ± 7,86 | **76,29 ± 2,94** |
| M3 CNN+Transformer | 62,00 ± 1,71 | **81,43 ± 2,86** |

Pri dotrénovaní mal model už po 5 epochách train accuracy 95–100 %, ale validačná zostala okolo 60 % a validačná loss rástla od 3. epochy. Na 361 videí je dotrénovanie CNN priveľa. Zamrznutá silnejšia CNN je lepšia aj stabilnejšia a trénovanie hlavy trvá sekundy. Výsledky prvej verzie sú v `Results/ablation_v1_convnext_tiny_finetuned/`. Kód tejto verzie už v projekte nie je, finálny kód CNN nedotrénuje.

## GradCAM

`Results/M2_CNN_LSTM/gradcam_run1.png`, `Results/M3_CNN_Transformer/gradcam_run1.png`: 3 správne a 3 nesprávne klasifikované testovacie videá, 8 snímok z každého. Skóre triedy za celý klip sa spätne šíri cez LSTM/Transformer do posledného stupňa CNN v každej snímke. Mapy sú normalizované cez celý klip, takže je vidieť nielen *kde*, ale aj *kedy* sa model pozerá. Pre M1 (RBF SVM) GradCAM nemá zmysel, lebo z SVM nevedie gradient ku skóre triedy.

Pozorovania (M3): pri *eat* sa model pozerá na ruku pri ústach a iba v snímkach, kde sa to deje. Pri *clap* sa pozerá na ruky. Pri chybe *drink → eat* sa pozerá na stôl s jedlom, teda rozhoduje sa podľa kontextu scény a nie podľa samotnej činnosti.

## Kamera

- `camera/demo.py`: buffer posledných 2,5 s z webkamery. Každých 0,25 s sa vyberie 16 rovnomerne rozložených snímok a klasifikujú sa (asi 56 ms na RTX 4060). Pravdepodobnosti sa vyhladzujú (EMA) a pod hranicou 50 % sa zobrazí „?“. Model pozná iba 4 činnosti, inak by vždy jednu z nich vypísal. Prepínač `--source video.avi --no-display` slúži na test bez kamery.
- `camera/record_clips.py`: nahratie vlastných 3 s klipov pre každú triedu (MEDZERNÍK → odpočet → nahrávanie) do `data/camera/<trieda>/`.
- `camera/finetune_camera.py`: rozdelí vlastné klipy pre každú triedu na 60 % train a 40 % test. Najlepší model otestuje na kamerových testovacích klipoch aj na HMDB teste. Potom ho dotrénuje na HMDB train + kamerových train klipoch (pri M1 nanovo natrénuje SVM s rovnakými C/gamma, pri M2/M3 10 epoch hlavy s 3× nižším lr). Nakoniec oba testy zopakuje a ukáže porovnanie pred a po (`Results/camera_finetune/<model>/before_after.csv`).

**Ešte treba urobiť:** nahrať vlastné klipy (aspoň 10 na triedu, ideálne viac ľudí, prostredí a osvetlení), spustiť `finetune_camera.py` a otestovať `demo.py` naživo.
