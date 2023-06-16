# Instrukcja uruchomienia

Aby uruchomić eksperymenty należy pobrać wymagane biblioteki przy użyciu polecenia:
```
pip install nazwa_biblioteki
```
Aby zainstalować framework COCO należy:
- Zainstalować cocopp poleceniem `pip install cocopp`.
- Pobrać repozytorium COCO `git clone https://github.com/numbbo/coco.git`.
- W folderze coco repozytorium wykonać polecenia: `python do.py build-python`, a następnie `python do.py run-python`.

W cenu wykonania eksperymentów należy wykonać polecenie:
```
python experiment.py
```
W celu wyświetlenia wyników eksperymentów należy otworzyć plik `ppdata/index.html`