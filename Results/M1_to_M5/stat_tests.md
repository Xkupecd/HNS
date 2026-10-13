# Štatistické testy testovacej accuracy (3 behy na model)

## zamrznutý backbone

Jednocestná ANOVA (počet modelov: 5): F = 66.66, p = 3.59e-07

Párové Welchove t-testy (p-hodnoty):

|    |       M1 |       M2 |       M3 |       M4 |       M5 |
|:---|---------:|---------:|---------:|---------:|---------:|
| M1 | nan      |   0.0201 |   0.0006 |   0.0005 |   0.0005 |
| M2 |   0.0201 | nan      |   0.3681 |   0.0367 |   0.2229 |
| M3 |   0.0006 |   0.3681 | nan      |   0.0130 |   0.0171 |
| M4 |   0.0005 |   0.0367 |   0.0130 | nan      |   0.0003 |
| M5 |   0.0005 |   0.2229 |   0.0171 |   0.0003 | nan      |

## postupné rozmrazovanie

Jednocestná ANOVA (počet modelov: 5): F = 6.26, p = 8.64e-03

Párové Welchove t-testy (p-hodnoty):

|    |       M1 |       M2 |       M3 |       M4 |       M5 |
|:---|---------:|---------:|---------:|---------:|---------:|
| M1 | nan      |   0.3693 |   0.0541 |   0.0085 |   0.5996 |
| M2 |   0.3693 | nan      |   0.1033 |   0.0569 |   0.2932 |
| M3 |   0.0541 |   0.1033 | nan      |   0.1763 |   0.1019 |
| M4 |   0.0085 |   0.0569 |   0.1763 | nan      |   0.0157 |
| M5 |   0.5996 |   0.2932 |   0.1019 |   0.0157 | nan      |

Poznámka: pri 3 behoch na model majú testy nízku silu; p < 0,05 berieme ako
štatisticky významný rozdiel. Ak majú oba modely nulový rozptyl, t-test nie je definovaný (nan).