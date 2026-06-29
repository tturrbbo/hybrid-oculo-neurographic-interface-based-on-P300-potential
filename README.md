# EyeGaze P300 экспериментальный стенд

python -m eyegaze.app.gaze_tiles_test --participant P01 --config config/experiment.yaml

python -m eyegaze.app.hybrid_gaze_p300_runner --participant P01 --config config/experiment.yaml 
## Установка

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск мониторингазз

```bash
python -m eyegaze.app.monitor
```

Показывает:
- точку взгляда;
- ячейку 3x3;
- yaw / pitch / roll;
- расстояние до камеры;
- статус ближе / дальше / норма.

## Запуск эксперимента

```bash
python -m eyegaze.app.experiment --participant P01
```

Логика:
1. калибровка по центру и 9 точкам;
2. случайный порядок 10 положений головы;
3. 5 секунд инструкция;
4. 15 секунд запись;
5. случайные плитки 3x3;
6. CSV сохраняется в `data/logs`.

## Анализ

```bash
python -m eyegaze.app.analyze
```

Результаты сохраняются в `data/results`.

## Файлы запуска Windows

- `START_MONITOR.bat`
- `START_EXPERIMENT.bat`
- `START_ANALYZE.bat`
