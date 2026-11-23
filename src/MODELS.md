# Получение предсказаний моделей

`predict.py` запускает одну модель локально, читает фотографии из manifest и записывает JSONL для `evaluate.py`. Режим `--contest` читает JSONL со ссылками на фотографии и создаёт файл для отправки в Контест. Эталон и таксономические категории модели не передаются. Одна команда загружает одну модель; четыре команды ниже выполняются последовательно.

## Где находятся модели

| Параметр | Веса и официальные примеры |
| --- | --- |
| `--model dino` | [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) |
| `--model owlv2` | [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) |
| `--model qwen` | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| `--model yolo-world` | [YOLO-Worldv2-S, веса `yolov8s-worldv2.pt`](https://docs.ultralytics.com/models/yolo-world/) |

Первые три модели Transformers скачивает при первом запуске и затем использует из локального кэша Hugging Face. YOLO-World загружается через Ultralytics; его веса по умолчанию сохраняются в `/weights/yolov8s-worldv2.pt`. Во время распознавания изображения остаются на машине, где выполняется Python; inference API не вызывается.


## Установка и пробный запуск

Из корня рабочей папки, в отдельном окружении Python (рекомендуется 3.12):

```bash
python3 -m venv .venv-models
source .venv-models/bin/activate
python -m pip install -r requirements-models.txt
python predict.py --model dino --limit 3 --output /predictions/smoke_dino.jsonl
python predict.py --model yolo-world --limit 3 --output /predictions/smoke_yolo-world.jsonl
```

Первый запуск скачивает веса, поэтому нужны интернет и свободное место. Доступ к публичным весам обычно не требует токена. Для NVIDIA PyTorch должен поддерживать установленный CUDA-драйвер; при необходимости используйте инструкцию установки на [сайте PyTorch](https://pytorch.org/get-started/locally/).

Зависимости включают Ultralytics 8.4.215 и CLIP из репозитория Ultralytics; для установки CLIP нужен Git. При задании классов YOLO-World также скачивает кодировщик CLIP ViT-B/32 и вычисляет признаки текстов один раз при загрузке.

`--device auto` выбирает CUDA, затем MPS на Apple Silicon, затем CPU. Можно явно указать `--device cpu`, `--device mps` или `--device cuda`. MPS здесь — предусмотренный вариант запуска, а не обещание совместимости каждой операции на любой версии macOS. Если на MPS возникает ошибка операции, повторите пробный запуск с `--device cpu`. Для DINO отключены дополнительные пользовательские CUDA-ядра; используется реализация PyTorch.

Детекторы используют float32; Qwen — float16 на CUDA/MPS и float32 на CPU. Квантование в этой версии скрипта не включено. Первые прогоны всех четырёх моделей выполнены на MPS: у DINO, OWLv2 и YOLO-World по 400 успешных ответов, у Qwen — 399 и одна ошибка JSON.

## Полный запуск и оценка

После успешной проверки нескольких изображений:

```bash
python predict.py --model dino
python predict.py --model owlv2
python predict.py --model qwen
python predict.py --model yolo-world
```

По умолчанию используются все 400 записей из `images_manifest.json`. Результаты появятся в `predictions/dino.jsonl`, `owlv2.jsonl`, `qwen.jsonl`, `yolo-world.jsonl`. Повторный запуск заменяет файлы с тем же именем; для экспериментов задавайте другой `--output`.

```bash
python evaluate.py --predictions predictions/dino.jsonl --manifest images_manifest.json --output results/dino.json
python evaluate.py --predictions predictions/owlv2.jsonl --manifest images_manifest.json --output results/owlv2.json
python evaluate.py --predictions predictions/qwen.jsonl --manifest images_manifest.json --output results/qwen.json
python evaluate.py --predictions predictions/yolo-world.jsonl --manifest images_manifest.json --output results/yolo-world.json
```


Для локальных изображений можно передать другой JSON manifest из записей `{"photo_id":123,"file":"images/123.jpg"}` через `--manifest path/to/manifest.json`. Пути `file` отсчитываются от каталога manifest. JSONL со ссылками на оригиналы поддерживается напрямую, как описано ниже.

## Подготовка ответа для Контеста

Чтобы скачать изображения отдельно и быстрее, рекомендуется использовать загрузчик:

```bash
python download_contest.py --workers 8
```

Он проверяет существующие фотографии полным декодированием, скачивает недостающие оригиналы и сохраняет размеры, SHA-256 и ссылки в `contest/download_report.json`. При временных сетевых ошибках выполняет до трёх попыток. Модели не загружаются. После завершения `predict.py` использует тот же кэш `contest/images`.

Пример для Qwen с ревизией весов и параметрами предыдущего прогона, из корня рабочей папки:

```bash
source .venv-models/bin/activate
python predict.py --contest --model qwen \
  --manifest contest/dataset.jsonl \
  --device mps \
  --max-new-tokens 1024 \
  --max-pixels 1048576 \
  --output contest/submission-qwen.jsonl
```

Рядом с итоговым файлом сохраняются:

- `submission-qwen.details.jsonl` — последний результат для каждого фото, рамки, confidence, суммарное время попыток и технические ошибки;
- `submission-qwen.raw.jsonl` — последние исходные ответы модели;
- `submission-qwen.attempts.jsonl` — исходный ответ, результат, ошибка и время каждой попытки;
- `submission-qwen.run.json` — параметры, ревизия весов, время скачивания/загрузки, признак `submission_ready` и история запусков в `previous_runs`.


## Повторить только ошибки или продолжить прерванный запуск

Укажите тот же `--output`, что в исходном запуске, и добавьте `--retry-errors`:

```bash
python predict.py --retry-errors \
  --output contest/submission-qwen.jsonl
```

Команда работает и тогда, когда файл `submission-qwen.jsonl` ещё не был создан из-за ошибок: исходные ответы берутся из `.details.jsonl`, настройки — из `.run.json`. Модель, ревизия весов, устройство, инструкция и параметры восстанавливаются автоматически. Сохраните служебные файлы рядом с выбранным `--output`.

Скрипт обрабатывает только фотографии с `error != null`, `boxes: null` или ещё без ответа. Успешные ответы, в том числе `boxes: []`, сохраняются без повторного вычисления. Модель загружается один раз для выбранных фотографий; при отсутствии ошибок и пропусков она вообще не загружается, а файл для Контеста собирается из сохранённых результатов.

Если исходный ответ Qwen зациклился на одинаковых рамках, можно повторить ошибочные фотографии со штрафом за повторы:

```bash
python predict.py --retry-errors \
  --output contest/submission-qwen.jsonl \
  --repetition-penalty 1.1
```

Параметр `repetition_penalty` применяется при генерации; значение `1.0` оставляет исходное поведение, значение выше `1.0` штрафует повторяющиеся токены. Это настройка модели, которая может изменить и состав рамок. [Документация Transformers](https://huggingface.co/docs/transformers/v4.57.0/main_classes/text_generation).

## Выходные файлы и ошибки

В обычном режиме, без `--contest`, сохраняются четыре файла:

- `<name>.jsonl`: `photo_id`, нормированные `boxes`, `scores`, `elapsed_seconds`, `error`. `[]` — успешный ответ без объектов. У Qwen нет сопоставимых detector confidence, поэтому его `scores` содержат `null`.
- `<name>.raw.jsonl`: координаты и оценки, возвращённые адаптером детектора, либо полный текст ответа Qwen, а также ошибка обработки. Для YOLO координаты уже прошли встроенный NMS и пересчитаны в пиксели исходного изображения; это не сырые выходы нейросети до NMS.
- `<name>.run.json`: выбранные веса, параметры запуска, настройки обработки изображений, версии библиотек, устройство, список ID и состояние завершения. Для моделей Hugging Face сохраняется commit SHA; для YOLO — путь и SHA-256 файла весов, параметры `yolo_predict` и сведения об установленной ревизии CLIP в `clip_installation`.
- `<name>.attempts.jsonl`: по одной записи на попытку с исходным ответом, результатом и временем. При `--retry-errors` журнал дополняется.

Время `elapsed_seconds` включает чтение фотографии, предобработку, вычисление модели с синхронизацией устройства и преобразование рамок. При повторе ошибки оно суммируется со временем предыдущих попыток этой фотографии; время отдельной попытки записано в `result.elapsed_seconds` журнала `.attempts.jsonl`. Поле `attempts` в текущем результате содержит число попыток. Загрузка весов учитывается отдельно в `load_seconds`; у YOLO сюда также входит подготовка текстовых классов. Предыдущие загрузки отражены в `previous_runs`. Первый снимок может выполняться медленнее из-за инициализации и прогрева предсказателя Ultralytics.

Если модель или разбор ответа завершились ошибкой, сохраняется `boxes: null` и текст ошибки. Ошибка не превращается в `[]`; скрипт продолжает обработку остальных кадров и в конце возвращает код 1. Частичные результаты записываются после каждого кадра; для продолжения используйте `--retry-errors` с прежним `--output`.

## Настройки эксперимента

- DINO, OWLv2 и YOLO-World получают запросы `butterfly` и `moth`. Изменить их можно через `--labels butterfly` или `--labels butterfly moth`. Исходный порог уверенности — `--threshold 0.25`. У DINO есть отдельный `--text-threshold 0.25`.

- Qwen получает инструкцию о взрослых бабочках, усиках, лапках, перекрытиях и пустых ответах. Текст сохранён в `QWEN_PROMPT`; заменить его можно через `--prompt '...'`, сохранив формат ответа. Генерация использует `do_sample=False`, лимит 1024 новых токенов и ограничение изображения примерно одним мегапикселем; параметры — `--max-new-tokens` и `--max-pixels`. Штраф повторов задаётся через `--repetition-penalty`, по умолчанию `1.0`. В исходном ответе сохраняются `generated_tokens` (включая EOS, если он сгенерирован) и `finish_reason`: `eos`, `length` или `unknown`. Достижение лимита без EOS считается ошибкой, исходный текст сохраняется целиком; незаконченный JSON автоматически не дописывается.

- YOLO-World использует вход `--imgsz 640`: изображение масштабируется с сохранением пропорций и дополняется полями до квадрата (`rect=False`). Встроенный NMS включён: `--yolo-nms-iou 0.5`, `agnostic_nms=True`; оба текстовых класса считаются одним целевым классом. Максимум — 300 рамок на снимок. `--yolo-nms-iou 1.0` отключает подавление по IoU, но оставляет порог confidence и лимит числа рамок. Все эти настройки сохраняются в `.run.json`.

- Все рамки приводятся к границам изображения, рамки нулевой площади удаляются. Дополнительный общий NMS по умолчанию выключен; включается через `--nms-iou 0.5`. Он объединяет все запросы в один класс и оставляет более уверенную рамку. Для Qwen, у которого нет confidence, при совпадениях сохраняется более ранняя рамка. У YOLO этот шаг выполняется после встроенного NMS; для обычного запуска достаточно `--yolo-nms-iou`.

- Для повторения запуска моделей Hugging Face передайте `--revision <resolved_revision>` из соответствующего `.run.json`. Для YOLO используйте тот же файл через `--yolo-weights path/to/yolov8s-worldv2.pt` и сверяйте `weights_sha256`; `--revision` к нему не применяется. Используйте записанные версии библиотек и параметры; ревизия Git-зависимости CLIP указана в `clip_installation.vcs_info.commit_id`.

## Четвёртая модель: YOLO-Worldv2-S

Для дополнительного сравнения добавлена **YOLO-Worldv2-S**, веса `yolov8s-worldv2.pt`, параметр `--model yolo-world`. Это детектор с открытым словарём классов: нужные объекты задаются текстом через `set_classes`, без обучения весов. Реализация Ultralytics публикует готовый вариант S и более крупные M/L/X. [Официальная документация](https://docs.ultralytics.com/models/yolo-world/).

Модель представляет другое семейство по сравнению с Grounding DINO, OWLv2 и Qwen. Её авторы ориентируются на быстрый поиск объектов по текстовым описаниям. Поэтому она полезна для сравнения качества и времени обработки. [Репозиторий авторов](https://github.com/AILab-CVC/YOLO-World).

**Гипотеза до эксперимента:** вариант S даст небольшое время обработки, но может уступить в полноте обнаружения мелких или закрытых бабочек и точности рамок вокруг усиков. Это предположение, а не измеренный результат на нашем датасете. Проверить F1@0.5, F1@0.75, пропуски, ложные срабатывания на 124 отрицательных изображениях и время на тех же 400 фотографиях.

Адаптер вызывает `set_classes(["butterfly", "moth"])`: стандартный словарь готовой модели содержит классы COCO. Полученные пиксельные `xyxy` приводятся к нормированным `left/top/width/height` и сохраняются в том же формате JSONL, что и у остальных моделей. [Реализация постобработки](https://github.com/ultralytics/ultralytics/blob/v8.4.215/ultralytics/models/yolo/detect/predict.py).

Для сравнения рассмотрен также [Florence-2-base](https://huggingface.co/microsoft/Florence-2-base), поддерживающий поиск областей по тексту. Для четвёртого эксперимента выбран YOLO-World, чтобы добавить компактный детектор с числовыми оценками уверенности и проверить компромисс между скоростью и качеством.
