# Домашнее задание 1

Утва Дмитрий Алексеевич

Датасет: [Amazon Reviews 2018, Musical Instruments](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html).

Код, графики и результаты: [experiments.ipynb](experiments.ipynb). Пять моделей реализованы в `models/`.

Запуск (Python 3.12):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_notebook.py
```

Данные скачиваются автоматически в `.data/` (236 МБ). Проверки: `python -m pytest -q`.
