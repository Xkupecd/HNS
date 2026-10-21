# TODO – čo ešte treba dorobiť v Projekte 2

Stav ku dňu 21. 10. 2026.

**Hotové:** dataset (HMDB51, 4 triedy, split 70/15/15 podľa zdrojového videa), 3 metódy (M1 CNN+SVM, M2 CNN+LSTM, M3 CNN+Transformer), 5 trénovaní každej, metriky a kontingenčné tabuľky, porovnanie, GradCAM, ablačná štúdia (prvá verzia s dotrénovaním CNN) a skripty pre kameru. Všetko je popísané v `README.md`.

**Chýba:** reálny test na kamere a dotrénovanie na kamere (to musíš spraviť ty, potrebuješ webkameru a seba pred ňou), dokumentácia, prezentácia a odovzdanie.

Termíny podľa harmonogramu: **prezentácia projektu 2 v 11. týždni** (22 b, dvojice), **odovzdanie do AIS do 12. týždňa**.

Všetky príkazy sa spúšťajú z priečinka `Projekt2/`:

```bash
cd ~/9.SEMESTER/HNS/Projekt2
P=../Projekt1/venv/bin/python
```

---

## A. Povinné podľa zadania

### 1. Nahrať vlastné klipy z kamery

Zadanie: *„kvalitu najlepšieho modelu reálne otestujte na kamere a dotrénujte model pre zlepšenie výsledkov na kamere“*.

```bash
$P -m src.camera.record_clips --person dominik --n 15
$P -m src.camera.record_clips --person <meno_spoluriesitela> --n 15
```

- [ ] Aspoň **10–15 klipov na triedu a osobu** (4 triedy × 15 = 60 klipov, asi 10 minút). Ovládanie: MEDZERNÍK = odpočet 3-2-1, potom 3 s nahrávania, `s` = ďalšia trieda, `q` = koniec. Klipy sa ukladajú do `data/camera/<trieda>/<osoba>_<trieda>_NN.avi`.
- [ ] Ak sa dá, nahrávajte **obaja** (projekt je pre dvojice), pri **rôznom osvetlení a pozadí** a v rôznej vzdialenosti od kamery. Čím rozmanitejšie klipy, tým dôveryhodnejší test.
- [ ] Činnosti predvádzajte tak, ako vyzerajú v HMDB51, lebo model sa učil z filmov:
  - **drink**: so skutočným pohárom alebo fľašou, zdvihnúť ich k ústam a napiť sa,
  - **eat**: so skutočným jedlom (rožok, jablko, lyžica), odhryznúť a žuť,
  - **wave**: mávanie rukou, ruka nech je celá v obraze (je to najslabšia trieda),
  - **clap**: tlieskanie rukami pred hrudníkom.
- [ ] Skontrolovať pár nahratých `.avi` súborov (prehrať ich), či majú 3 s a nie sú čierne.

### 2. Test na kamere a dotrénovanie

```bash
$P -m src.camera.finetune_camera                 # najlepší model podľa priemeru = M1 CNN+SVM
$P -m src.camera.finetune_camera --model m3      # odporúčam aj M3 (má najlepší jednotlivý beh)
```

- [ ] Spustiť pre **M1** (najlepší podľa priemeru) a ideálne aj pre **M3**.
- [ ] Pozrieť `Results/camera_finetune/<model>/before_after.csv`. Tabuľka ukazuje accuracy na kamerových testovacích klipoch **pred** a **po** dotrénovaní a kontrolu, či model nezabudol HMDB (`hmdb_test`). Kontingenčné tabuľky sú v podpriečinkoch `before_*` a `after_*`.
- [ ] Ak je kamerových klipov málo (napr. 10 na triedu → 4 testovacie), výsledok je veľmi zašumený. Vtedy treba nahrať ďalšie klipy (skript pokračuje v číslovaní) a spustiť to znova.
- [ ] Ak dotrénovanie výrazne zhorší `hmdb_test`, skúsiť `--epochs 5` (M2/M3).

### 3. Živý test na kamere

```bash
$P -m src.camera.demo                                     # pôvodný najlepší model (M1)
$P -m src.camera.demo --model m1 --checkpoint Results/camera_finetune/M1_CNN_SVM/finetuned_svm.joblib
$P -m src.camera.demo --model m3 --checkpoint Results/camera_finetune/M3_CNN_Transformer/finetuned.pt
```

- [ ] Overiť, že sa otvorí okno s obrazom z kamery a stĺpcami pravdepodobností (`q` = koniec). Ak sa kamera neotvorí, skúsiť `--camera 1` (v systéme je `/dev/video0`, `video1` aj `video10`).
- [ ] Porovnať model **pred a po** dotrénovaní naživo a zapísať si dojmy (ktoré triedy fungujú, ktoré sa pletú). Pozor: medzi činnosťami model vždy niečo ukáže, lebo pozná iba 4 triedy. Pri neistote pod 50 % zobrazí „?“.
- [ ] **Nahrať krátke video z dema pre prezentáciu:** `$P -m src.camera.demo --save demo_kamera.mp4` (každú činnosť 2–3×). Zíde sa aj ako záloha, keby na prezentácii nefungovala kamera.
- [ ] Doplniť výsledky z kamery do `README.md` (sekcia „Kamera“): tabuľku pred/po z `before_after.csv` a pár viet o živom teste.

### 4. Dokumentácia

Odporúčam rovnakú štruktúru a šablónu (FEIstyle) ako v Projekte 1 (`../Projekt1/Documentation_Latex/`). Môžem ju pripraviť, keď budú hotové body 1–3.

- [ ] **Úvod a zadanie:** skupina 9, tabuľka „požiadavka zadania → kde je splnená“.
- [ ] **Teória:** CNN, transfer learning, SVM (RBF), LSTM, Transformer/self-attention, GradCAM, metriky (accuracy, precision, recall, F1, senzitivita, špecificita).
- [ ] **Dáta:** HMDB51, výber 4 tried (dajú sa predviesť pred kamerou), počty videí, **split podľa zdrojového videa** a prečo (`The_Slow_Clap` má 31 klipov), extrakcia snímok, augmentácia na úrovni klipu.
- [ ] **Metodika:** zamrznutý ConvNeXt-B, cache príznakov, 3 časové modely, hyperparametre, výber checkpointu, 5 behov.
- [ ] **Výber backbonu:** tabuľka 9 kandidátov podľa validačnej accuracy (z `README.md`).
- [ ] **Výsledky M1, M2, M3:** priebehy učenia (`Results/M*/best_run_training_curves.png`), kontingenčné tabuľky (`best_run_confusion_matrix.png`), tabuľky metrík, pri M1 grid search (`run*/grid_search.csv`).
- [ ] **Porovnanie:** `Results/M1_to_M3/metrics_bar.png`, `accuracy_per_run.png`, `comparison.md`, `per_class.csv`.
- [ ] **Ablácia:** prvá verzia s dotrénovaným ConvNeXt-T (58–62 %) proti finálnej (76–83 %), preučenie (`Results/ablation_v1_convnext_tiny_finetuned/*/run1/training_curves.png`).
- [ ] **GradCAM:** `Results/M2_CNN_LSTM/gradcam_run1.png`, `Results/M3_CNN_Transformer/gradcam_run1.png`, interpretácia (ruka pri ústach pri *eat*, kontext stola pri chybe *drink → eat*).
- [ ] **Kamera:** postup, tabuľka pred/po dotrénovaní, kontingenčné tabuľky, snímky z dema.
- [ ] **Záver a diskusia:** M1 ≈ M3 (rozdiel v rámci šumu, 1 testovacie video = 1,4 p. b.), prečo poradie snímok pri tak malom datasete nepomohlo, slabá trieda *wave*, obmedzenia (iba 70 testovacích videí, filmové scény ≠ webkamera).
- [ ] **Hardvér:** RTX 4060 Laptop 8 GB. Časy: cache príznakov asi 3,5 min, M1 asi 30 s na beh, M2/M3 asi 3 s na beh.
- [ ] **Titulná strana:** obaja riešitelia, akademický rok 2026/2027.

### 5. Prezentácia (11. týždeň)

- [ ] Snímky (zadanie, dáta a split, metódy, výsledky, ablácia, GradCAM, kamera, záver), Beamer ako pri Projekte 1.
- [ ] **Živé demo na kamere** + záložné video z bodu 3. Zobrať pohár a niečo na jedenie.
- [ ] Poznámky k snímkam a odpovede na pravdepodobné otázky (prečo zamrznutá CNN, prečo SVM vyhralo, prečo split podľa zdroja).

### 6. Odovzdanie do AIS (do 12. týždňa)

- [ ] `src/`, `README.md`, `requirements.txt`, `data_split/manifest.csv`, `Results/` a PDF dokumentácie.
- [ ] **Nedávať dataset** (`data/hmdb51`, `data/frames`, `data/features`, spolu asi 2,5 GB). V README je odkaz na stiahnutie a postup, ako dáta pripraviť.
- [ ] `Results/` má asi 230 MB. Checkpointy sú malé (hlava M2/M3 asi 5 MB, `svm.joblib` asi 17 MB). Ak má AIS limit, vynechať `svm.joblib` z `Results/ablation_v1_*` (tie sa už nepoužívajú) a prípadne checkpointy behov, ktoré nie sú najlepšie (`best_run` v `summary.json`).

---

## B. Odporúčané – rýchle vylepšenia (bez nového trénovania)

- [ ] **Výberová smerodajná odchýlka.** `src/evaluation.py` (`summarize_runs`) počíta `values.std()` (ddof = 0). Pri 5 behoch sa zvyčajne uvádza výberová, `values.std(ddof=1)`, ktorá je o 12 % väčšia. Po zmene spustiť znova `train_m1_cnn_svm` (30 s na beh), `train_m2/m3` (sekundy) a `compare_models`. Rovnako to treba zjednotiť s dokumentáciou Projektu 1.
- [ ] **Štatistický test M1 vs. M3.** Napr. Mann-Whitneyho test (5 vs. 5 behov) alebo McNemarov test na predikciách najlepších behov. Doplní argument, že rozdiel nie je významný.
- [ ] **Popisky grafov po slovensky.** GradCAM a porovnávacie grafy majú anglické popisky (`src/gradcam.py`, `src/compare_models.py`, `src/plots.py`).
- [ ] **Vysypať kôš.** `~/.local/share/Trash/files/raw/` má 2,1 GB (pôvodné `.rar` archívy) a na disku je voľných iba asi 24 GB.

## C. Voliteľné – rozšírenia (bonusové body, vyžadujú trénovanie)

- [ ] **Trieda „nič / iná činnosť“** pre kameru: nahrať klipy, kde človek iba sedí, hovorí alebo sa hýbe inak, a pridať 5. triedu (napr. z HMDB `sit`, `stand`, `talk`). Demo by potom nemuselo stále hádať jednu zo 4 činností.
- [ ] **Videomodel predtrénovaný na Kinetics** (napr. `torchvision.models.video.r2plus1d_18` alebo X3D) ako 4. metóda. Kinetics obsahuje aj triedy tlieskania, pitia a jedenia, takže by mal byť výrazne lepší. Bolo by to zaujímavé porovnanie „2D CNN + časový model“ proti „3D CNN“.
- [ ] **Viac splitov (krížová validácia po skupinách).** Test má iba 70 videí, takže 5 rôznych group-splitov by dalo spoľahlivejší odhad ako 5 behov na jednom splite.
- [ ] **Rýchlejšie demo:** počítať príznaky CNN pre každú snímku iba raz, keď príde, a nie znova pre celé okno každých 0,25 s. Dnes to trvá asi 56 ms na predikciu, pomohlo by to hlavne na slabšom počítači.
