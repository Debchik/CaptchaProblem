# Спецификация фичей и пайплайна для соревнования CAPTCHA Bot Detection

## Цель

Построить сильный классификатор для бинарной задачи:

- `target = 0` — бот;
- `target = 1` — человек.

Основная метрика: `roc_auc_score(..., max_fpr=0.1)`.  
Это значит, что особенно важно хорошо отделять ботов в зоне малого false positive rate. Модель должна не просто давать хороший общий AUC, а уметь уверенно находить ботов, не выбрасывая слишком много людей.

Данных мало по labels:

- `train.parquet`: 1000 размеченных объектов;
- `test.parquet`: 100000 объектов без target;
- `unlabeled.parquet`: 200000 объектов без target.

Главная стратегия: использовать `train` как источник меток, а `unlabeled + test` как источник структуры поведения, шаблонов и распределений.

Итоговая модель должна обучаться на комбинации:

1. ручных behavioral-forensics фичей;
2. физических фичей движения;
3. энтропийных и символьных фичей;
4. template-mining / duplicate-mining фичей;
5. self-supervised trajectory embeddings;
6. pseudo-labeling только для очень уверенных объектов.

---

## Общий формат разработки

Все блоки можно реализовывать параллельно. Главное требование: каждый блок должен возвращать `DataFrame` с колонкой `id` или индексом, совпадающим с исходным порядком объектов.

Для каждого датасета нужно уметь вызвать:

```python
features_train = make_features(train_df, dataset_name="train")
features_test = make_features(test_df, dataset_name="test")
features_unlabeled = make_features(unlabeled_df, dataset_name="unlabeled")
```

Итоговая сборка:

```python
X_train = concat_all_feature_blocks(train)
y_train = train["target"]
X_test = concat_all_feature_blocks(test)
```

Все признаки, использующие target через соседей, кластеры или hash-статистики, должны считаться для train строго out-of-fold. Иначе будет leakage.

---

## Базовый парсинг событий

Нужно реализовать единый парсер для `mouse_events` и `touch_events`.

### Mouse events

Ожидаемый формат элемента:

```python
{x_, y_, timestamp_}
```

Нужно извлекать:

- `x`
- `y`
- `timestamp`

### Touch events

Ожидаемый формат элемента:

```python
{x_, y_, timestamp_, force_, radiusX_, radiusY_, rotationAngle_}
```

Нужно извлекать:

- `x`
- `y`
- `timestamp`
- `force`
- `radiusX`
- `radiusY`
- `rotationAngle`

### Общая нормализация

Для каждой строки:

```python
x_norm = x / viewport_width
y_norm = y / viewport_height
t = timestamp - first_timestamp
```

События отсортировать по `timestamp`.

Для всех sequence-фичей нужны safe cases:

- если событий нет: вернуть флаги `has_mouse = 0` / `has_touch = 0` и заполнить агрегаты `-1` или `NaN`, затем обработать моделью;
- если событий меньше 2: не считать производные;
- если `dt <= 0`: пометить отдельным флагом и не использовать такие пары в физических производных;
- если `viewport_width <= 0` или `viewport_height <= 0`: поставить флаг невалидности.

---

## Блок A. Consistency / impossible behavior features

Этот блок самый быстрый и потенциально самый сильный. Он ловит технические артефакты ботов.

### A1. Count consistency

Для `mouse_events`:

```python
mouse_len_sampled = len(mouse_events)
mouse_events_total
mouse_total_minus_len = mouse_events_total - mouse_len_sampled
mouse_is_truncated = mouse_events_total > mouse_len_sampled
mouse_over_100 = mouse_events_total > 100
```

Для `touch_events` аналогично:

```python
touch_len_sampled
touch_events_total
touch_total_minus_len
touch_is_truncated
touch_over_100
```

Важно: по условию массив хранит максимум 100 последних событий. Если `events_total > 100`, начало поведения потеряно. Это отдельный сильный сигнал.

Дополнительно:

```python
mouse_saved_first_timestamp
mouse_saved_last_timestamp
mouse_saved_duration = last - first

touch_saved_first_timestamp
touch_saved_last_timestamp
touch_saved_duration
```

### A2. Timestamp validity

Для mouse и touch отдельно:

```python
num_non_monotonic_timestamps
num_duplicate_timestamps
min_dt
median_dt
mean_dt
std_dt
max_dt
dt_unique_count
dt_unique_ratio = dt_unique_count / len(dt)
share_dt_equal_50
share_dt_multiple_of_50
share_dt_less_50
share_dt_zero_or_negative
dt_mod_50_mean
dt_mod_50_std
```

Почему важно: события мыши записываются не чаще одного раза в 50 мс. Скрипты часто генерируют слишком регулярные интервалы.

### A3. Pointer event order

Использовать поля:

- `relative_captcha_init_time`
- `hover_timestamp`
- `pointerdown_timestamp`
- `pointerup_timestamp`

Фичи:

```python
init_to_hover_time = hover_timestamp - relative_captcha_init_time
hover_to_down_time = pointerdown_timestamp - hover_timestamp
down_to_up_duration = pointerup_timestamp - pointerdown_timestamp
init_to_down_time = pointerdown_timestamp - relative_captcha_init_time
init_to_up_time = pointerup_timestamp - relative_captcha_init_time

is_hover_before_init = hover_timestamp < relative_captcha_init_time
is_down_before_hover = pointerdown_timestamp < hover_timestamp
is_up_before_down = pointerup_timestamp < pointerdown_timestamp
is_negative_click_duration = down_to_up_duration < 0
is_zero_click_duration = down_to_up_duration == 0
```

### A4. Coordinate validity

Для mouse и touch:

```python
share_x_outside_viewport
share_y_outside_viewport
share_points_outside_viewport
num_points_outside_viewport
```

Для pointer events:

```python
pointerdown_x_norm = pointerdown_x / viewport_width
pointerdown_y_norm = pointerdown_y / viewport_height
pointerup_x_norm = pointerup_x / viewport_width
pointerup_y_norm = pointerup_y / viewport_height
hover_x_norm = hover_x / viewport_width
hover_y_norm = hover_y / viewport_height

pointerdown_outside_viewport
pointerup_outside_viewport
hover_outside_viewport
```

### A5. Coordinate quantization

Для mouse и touch отдельно:

```python
share_x_integer
share_y_integer
share_x_half_integer
share_y_half_integer
x_fraction_unique_count
y_fraction_unique_count
x_unique_ratio
y_unique_ratio
```

Дополнительно можно оценить грубую координатную сетку:

```python
median_abs_dx_nonzero
median_abs_dy_nonzero
min_abs_dx_nonzero
min_abs_dy_nonzero
```

Если координаты слишком часто целые, кратные одному шагу или лежат на сетке, это может быть бот.

### A6. Mouse/touch grammar

```python
has_mouse_events
has_touch_events
has_both_mouse_and_touch
has_no_movement
has_pointer_but_no_mouse_or_touch
has_mouse_without_pointer
has_touch_without_pointer
```

Расстояния от последних событий до pointer-событий:

```python
distance_last_mouse_to_pointerdown
distance_last_mouse_to_pointerup
distance_last_touch_to_pointerdown
distance_last_touch_to_pointerup
```

### A7. Touch-specific consistency

Для touch:

```python
force_mean
force_std
force_min
force_max
force_unique_count
force_unique_ratio
force_is_constant
share_force_zero
share_force_one

radiusX_mean
radiusX_std
radiusY_mean
radiusY_std
radius_ratio_mean = mean(radiusX / (radiusY + eps))
radius_ratio_std
radius_is_constant

rotationAngle_mean
rotationAngle_std
rotationAngle_unique_count
rotationAngle_is_constant
share_rotation_zero
```

---

## Блок B. Физика движения

Этот блок описывает человеческую моторику: скорость, ускорение, jerk, кривизну, коррекции.

Реализовать одинаково для mouse и touch. Префиксы колонок: `mouse_...`, `touch_...`.

### B1. Производные движения

Для соседних валидных точек:

```python
dx = x[i] - x[i-1]
dy = y[i] - y[i-1]
dt = t[i] - t[i-1]

dist = sqrt(dx**2 + dy**2)
speed = dist / dt
angle = atan2(dy, dx)
turn = wrapped_angle_diff(angle[i], angle[i-1])
acceleration = diff(speed) / dt[1:]
jerk = diff(acceleration) / dt[2:]
```

Использовать `eps` для деления. Для сильно выбивающихся значений лучше считать и raw, и clipped/log версии:

```python
log_speed = log1p(speed)
log_dt = log1p(dt)
log_abs_acc = log1p(abs(acceleration))
log_abs_jerk = log1p(abs(jerk))
```

### B2. Геометрия траектории

```python
path_len = sum(dist)
straight_dist = distance(first_point, last_point)
efficiency = straight_dist / (path_len + eps)
tortuosity = path_len / (straight_dist + eps)

bbox_width = max(x) - min(x)
bbox_height = max(y) - min(y)
bbox_area = bbox_width * bbox_height
bbox_aspect = bbox_width / (bbox_height + eps)

mean_deviation_from_start_end_line
max_deviation_from_start_end_line
std_deviation_from_start_end_line
```

Интерпретация:

- бот-линия: `efficiency` близко к 1, маленькое отклонение от линии;
- человек: больше микрокривизны и коррекций;
- рандомный бот: слишком большая tortuosity/хаотичность.

### B3. Скорость

```python
duration = t_last - t_first
mean_speed
median_speed
std_speed
max_speed
min_speed
speed_cv = std_speed / (mean_speed + eps)
speed_q10
speed_q25
speed_q75
speed_q90
speed_iqr = q75 - q25

peak_speed_position = argmax(speed) / len(speed)
time_to_peak_speed_ratio = timestamp_at_max_speed / duration
num_speed_peaks
speed_skew
speed_kurtosis
```

### B4. Acceleration и jerk

```python
mean_abs_acceleration
median_abs_acceleration
std_acceleration
max_abs_acceleration
acceleration_q90_abs
acceleration_sign_changes

mean_abs_jerk
median_abs_jerk
std_jerk
max_abs_jerk
jerk_q90_abs
jerk_energy = sum(jerk**2)
jerk_sign_changes
```

`jerk` важен: он ловит неестественно резкие или слишком идеально гладкие изменения движения.

### B5. Кривизна и повороты

```python
turn_abs_sum = sum(abs(turn))
turn_mean_abs
turn_median_abs
turn_std
turn_max_abs
turn_q90_abs
num_sharp_turns_30deg
num_sharp_turns_60deg
num_sharp_turns_90deg
num_turn_sign_changes
curvature_energy = sum(turn**2)
straight_segment_ratio = share(abs(turn) < small_threshold)
```

### B6. Minimum-jerk profile fit

Человеческое движение часто похоже на smooth minimum-jerk profile.

Нормализуем время:

```python
tau = (t - t0) / (t_last - t0 + eps)
```

Идеальный прогресс вдоль движения:

```python
s_ideal = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
```

Фактический прогресс:

```python
line = end_point - start_point
s_actual = projection(current_point - start_point, line) / (norm(line)**2 + eps)
```

Фичи:

```python
minjerk_mse
minjerk_mae
minjerk_max_error
minjerk_corr
```

### B7. Коррекции перед кликом / отпусканием

```python
distance_to_final[i] = distance(point[i], final_point)

num_times_distance_to_final_increases
share_distance_to_final_increases
last_third_path_len_ratio
last_third_turn_abs_sum
last_third_mean_speed
last_third_speed_std
endpoint_jitter = mean distance between last k points
```

Для человека часто есть микрокоррекции перед финальным нажатием.

### B8. Pointer geometry

```python
distance_hover_to_down
distance_down_to_up
distance_hover_to_up

distance_first_event_to_hover
distance_first_event_to_down
distance_last_event_to_down
distance_last_event_to_up
```

---

## Блок C. Энтропия и символьная динамика

Не считать энтропию сырых `x/y`. Нужно переводить движение в последовательность состояний.

### C1. Токенизация движения

Для каждой траектории построить:

```python
direction_bin = bin angle into 8 or 16 bins
speed_bin = quantile bin of log_speed into 5 bins
turn_bin = bin turn into categories
pause_bin = int(dt > threshold or dist == 0)
```

Пример turn categories:

```python
sharp_left: turn < -60 deg
left: -60 <= turn < -15
straight: -15 <= turn <= 15
right: 15 < turn <= 60
sharp_right: turn > 60
```

Собрать общий state-token:

```python
state = direction_bin + 16 * speed_bin + 16 * 5 * turn_bin + 16 * 5 * 5 * pause_bin
```

### C2. Shannon entropy

Для последовательностей:

- `direction_bin`
- `speed_bin`
- `turn_bin`
- `pause_bin`
- `state`

Посчитать:

```python
entropy_direction
entropy_speed
entropy_turn
entropy_pause
entropy_state
normalized_entropy_state = entropy_state / log(num_unique_states + eps)
unique_state_ratio = num_unique_states / len(sequence)
```

### C3. Transition entropy

Для пар `(state_t, state_{t+1})`:

```python
transition_entropy_state = H(state_next | state_current)
transition_unique_ratio
most_common_transition_share
```

То же можно сделать для direction:

```python
transition_entropy_direction
most_common_direction_transition_share
```

### C4. Runs / repeated patterns

```python
max_repeated_same_state_run
mean_repeated_same_state_run
same_state_run_ratio
max_same_direction_run
mean_same_direction_run
same_direction_run_ratio
```

### C5. Lempel-Ziv complexity

Для строк:

- direction sequence;
- speed_bin sequence;
- turn_bin sequence;
- state sequence.

Фичи:

```python
lz_complexity_direction
lz_complexity_speed
lz_complexity_turn
lz_complexity_state
normalized_lz_complexity_state = lz_complexity_state / len(sequence)
```

### C6. Permutation entropy

Для числовых рядов:

- speed;
- acceleration;
- turn.

Фичи:

```python
permutation_entropy_speed_order3
permutation_entropy_speed_order4
permutation_entropy_turn_order3
permutation_entropy_acceleration_order3
```

Если ряд короткий, возвращать `-1` и флаг валидности.

### C7. Энтропия по частям траектории

Разделить sequence на 3 части:

```python
first_third
middle_third
last_third
```

Посчитать:

```python
entropy_state_first_third
entropy_state_middle_third
entropy_state_last_third
entropy_turn_first_third
entropy_turn_last_third
```

Смысл: у человека начало, движение и финальная коррекция могут иметь разную структуру.

---

## Блок D. Template mining / duplicate mining

Цель: найти повторяющиеся семейства ботов в `train + unlabeled + test`.

### D1. Нормализованные представления траектории

Для каждой mouse/touch траектории сделать несколько векторных представлений.

#### Representation 1: viewport-normalized resampled

```python
x_norm = x / viewport_width
y_norm = y / viewport_height
resample trajectory to 32 or 64 points by time
vector = [x1, y1, ..., x64, y64]
```

#### Representation 2: start-end-normalized geometry

```python
x = x - x_start
y = y - y_start
scale = straight_dist or path_len
x = x / scale
y = y / scale
resample to 32 or 64 points
```

#### Representation 3: arc-length-resampled

Ресемплировать не по времени, а по пройденной длине пути.

#### Representation 4: symbolic n-gram representation

Использовать sequences:

- `direction_bin`
- `speed_bin`
- `turn_bin`
- `state`

Собрать n-grams длины 2, 3, 4. Можно использовать `TfidfVectorizer` по строкам токенов.

### D2. Hash features

Собрать hash-ключи:

```python
dir_hash
speed_hash
turn_hash
state_hash
dir_speed_hash
dir_turn_dt_hash
```

Для каждого hash по объединению `train + unlabeled + test` посчитать:

```python
hash_count_all
hash_count_train
hash_count_unlabeled
hash_count_test
```

Для train target-статистики считать out-of-fold:

```python
hash_bot_count_oof
hash_human_count_oof
hash_target_mean_oof
hash_target_std_oof
hash_label_entropy_oof
```

Для test/unlabeled target-статистики считать по всему train.

### D3. kNN duplicate features

Построить nearest-neighbor индексы на разных представлениях:

1. manual motion features;
2. resampled trajectory vector;
3. symbolic n-gram TF-IDF vector;
4. SSL embedding, когда появится.

Фичи:

```python
knn_dist_1
knn_dist_3_mean
knn_dist_5_mean
knn_dist_10_mean
knn_dist_20_mean
knn_dist_std_10

knn_train_dist_min
knn_unlabeled_dist_min
knn_test_dist_min
```

Target-based kNN фичи:

```python
knn_train_label_mean_3
knn_train_label_mean_5
knn_train_label_mean_10
knn_train_label_std_10
knn_train_bot_dist_min
knn_train_human_dist_min
knn_bot_human_dist_ratio = bot_dist_min / (human_dist_min + eps)
```

Для train — строго OOF. Для test — обучаем kNN на всем train.

### D4. Clustering features

На trajectory vectors и manual motion features запустить:

- `MiniBatchKMeans`, например `n_clusters = 50, 100, 200`;
- если успеваем: `HDBSCAN` или `DBSCAN`.

Фичи:

```python
cluster_id
cluster_size_all
cluster_size_train
cluster_size_unlabeled
cluster_size_test
distance_to_cluster_center
cluster_density_proxy
```

Target-based фичи, OOF для train:

```python
cluster_target_mean_oof
cluster_target_std_oof
cluster_label_entropy_oof
cluster_bot_count_oof
cluster_human_count_oof
```

---

## Блок E. Self-supervised 1D CNN / TCN encoder

Этот блок делается параллельно. Его не нужно ждать для первого CatBoost. Сначала модель на ручных фичах, потом добавить embeddings.

### E1. Данные для pretraining

Использовать:

```python
train + unlabeled
```

`test` можно использовать только если правила соревнования допускают transductive feature learning без target leakage. Если сомневаемся — сначала не использовать test.

### E2. Input tensor

Последовательность длины `seq_len = 100`. Если событий меньше — padding. Если больше — уже сохранены последние 100.

Каналы:

```python
x_norm
y_norm
t_norm
dx
dy
log_dt
distance
log_speed
acceleration
jerk
sin_angle
cos_angle
sin_turn
cos_turn
is_mouse
is_touch
is_padding
force
radiusX
radiusY
rotationAngle
```

Для отсутствующих touch-полей ставить 0 и использовать mask/channel flag.

### E3. Архитектура

Минимальная архитектура:

```python
Conv1d(C, 64, kernel_size=5, padding=2)
BatchNorm1d или LayerNorm-style нормализация
ReLU/GELU

Residual TCN block x 3-5
GlobalAvgPool + GlobalMaxPool
Linear -> embedding_dim=64 или 128
```

Не делать большую модель. Labels мало, а времени еще меньше.

### E4. Self-supervised task A: masked trajectory modeling

Случайно маскировать 15-30% валидных точек.

Модель должна восстановить:

```python
dx
dy
log_dt
speed_bin
direction_bin
turn_bin
```

Loss:

```python
HuberLoss(dx, dy, log_dt)
CrossEntropy(speed_bin, direction_bin, turn_bin)
```

### E5. Self-supervised task B: next-step prediction

Для каждого valid step предсказывать следующий state-token:

```python
state_{t+1}
```

Loss:

```python
CrossEntropy(next_state)
```

### E6. Self-supervised task C: contrastive learning

Сделать две аугментированные версии одной траектории.

Разрешенные аугментации:

```python
small coordinate jitter
time scaling
random crop
event dropout
resampling jitter
small viewport scaling
```

Нельзя:

```python
reverse trajectory
large rotation
shuffle events
large coordinate distortion
```

Loss:

```python
InfoNCE / NT-Xent
```

Если времени мало, сделать только masked modeling + next-step prediction.

### E7. Извлечение фичей

После pretraining:

```python
embedding_0 ... embedding_63
```

Дополнительно посчитать per-object losses:

```python
masked_reconstruction_loss
next_state_prediction_loss
```

Их тоже добавить в CatBoost. Высокая ошибка восстановления может означать необычное поведение.

### E8. Fine-tuning на labels

Основной безопасный вариант:

```python
freeze encoder
train small classification head on train
extract penultimate embeddings
```

Рискованный вариант:

```python
unfreeze last TCN block
small lr
early stopping
strong dropout
```

Использовать только если CV по `max_fpr=0.1` реально растет.

---

## Блок F. Pseudo-labeling

Pseudo-labeling делать только после сильной первой модели.

### F1. Получить out-of-fold train predictions и predictions для unlabeled

Модель: CatBoost/LightGBM ensemble на ручных + template + SSL фичах.

### F2. Выбрать только очень уверенные объекты

Так как `target=1` — человек, `target=0` — бот:

```python
pseudo_human = p_human > 0.995
pseudo_bot = p_human < 0.005
```

Если таких мало, ослабить до:

```python
p_human > 0.99
p_human < 0.01
```

Но не ниже без проверки CV.

### F3. Добавить с маленьким весом

```python
sample_weight_train = 1.0
sample_weight_pseudo = 0.2 или 0.3
```

Делать максимум 1-2 итерации. Больше — риск самозаражения модели.

---

## Итоговая модель

### Основной классификатор

Рекомендуемый первый вариант:

```python
CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    depth=3, 4 или 5,
    learning_rate=0.02-0.05,
    l2_leaf_reg=5-30,
    iterations=3000-8000,
    random_strength=1-5,
    auto_class_weights="Balanced",
    verbose=200
)
```

Почему CatBoost: мало данных, много heterogeneous фичей, устойчивость к странным распределениям.

### Дополнительные модели для ансамбля

Запустить параллельно:

```python
LightGBM
LogisticRegression on normalized selected features
CatBoost only manual features
CatBoost manual + template
CatBoost manual + template + SSL
```

Финальный prediction:

```python
pred = weighted_rank_average(preds)
```

Вес давать по CV `roc_auc_score(y, pred, max_fpr=0.1)`.

---

## Валидация

### Главная CV-схема

Из-за 1000 labels обычный split шумный. Использовать:

```python
RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=...)
```

Метрика:

```python
roc_auc_score(y_true, y_pred, max_fpr=0.1)
```

### Sanity check против leakage

Для template/hash/cluster фичей дополнительно проверить:

```python
GroupKFold by trajectory_hash or cluster_id
```

Если обычный CV сильно растет, а GroupKFold падает — модель выучила дубликаты train, а не поведение.

### Feature ablation

Проверить вклад блоков:

```python
base only
base + consistency
base + consistency + motion
base + consistency + motion + entropy
base + consistency + motion + entropy + template
all + SSL
all + pseudo-labeling
```

Нужно сохранять таблицу результатов.

---

## Приоритет реализации

Поскольку времени мало, разработчики должны работать параллельно по блокам.

### Developer 1: Parser + consistency

Сделать:

- парсер mouse/touch;
- timestamp фичи;
- pointer-order фичи;
- viewport validity;
- coordinate quantization;
- touch-specific consistency.

Выход:

```python
features_consistency_train.csv
features_consistency_test.csv
features_consistency_unlabeled.csv
```

### Developer 2: Motion physics

Сделать:

- geometry;
- speed;
- acceleration;
- jerk;
- curvature;
- min-jerk fit;
- endpoint correction features.

Выход:

```python
features_motion_train.csv
features_motion_test.csv
features_motion_unlabeled.csv
```

### Developer 3: Entropy / symbolic dynamics

Сделать:

- direction/speed/turn/state tokenization;
- Shannon entropy;
- transition entropy;
- runs;
- LZ complexity;
- permutation entropy.

Выход:

```python
features_entropy_train.csv
features_entropy_test.csv
features_entropy_unlabeled.csv
```

### Developer 4: Template mining

Сделать:

- trajectory resampling;
- hash counts;
- kNN distances;
- OOF target kNN features;
- cluster features.

Выход:

```python
features_template_train.csv
features_template_test.csv
features_template_unlabeled.csv
```

### Developer 5: SSL encoder

Сделать:

- tensor dataset для trajectories;
- small TCN encoder;
- masked modeling / next-step prediction;
- embeddings для train/test/unlabeled;
- reconstruction losses.

Выход:

```python
features_ssl_train.csv
features_ssl_test.csv
features_ssl_unlabeled.csv
```

### Developer 6: Modeling / validation / submission

Сделать:

- сборку всех feature blocks;
- CV с `max_fpr=0.1`;
- CatBoost/LightGBM;
- ablation table;
- ensemble;
- pseudo-labeling;
- submission file.

Формат submission:

```csv
id,prediction
0,0.95882830637116846
1,0.9888864313051369
...
```

`prediction` — вероятность класса `1`, то есть человека.

---

## Минимальный must-have набор, если времени совсем мало

Если не успеваем всё, обязательно сделать эти блоки:

```text
1. Consistency / impossible behavior
2. Motion physics: speed, jerk, curvature, geometry
3. Entropy: direction/speed/turn/state entropy
4. Template hash counts + kNN distances
5. CatBoost with repeated stratified CV
```

SSL encoder полезен, но не должен блокировать основной пайплайн.

---

## Главные риски

### 1. Leakage в target encoding

Любые фичи вида:

```python
hash_target_mean
cluster_target_mean
knn_label_mean
```

для train считать только out-of-fold.

### 2. Переобучение SSL на 1000 labels

Encoder сначала pretrain без target. Потом либо freeze, либо очень осторожный fine-tuning.

### 3. Неправильная обработка коротких траекторий

Для коротких sequences ставить validity flags. Не пытаться считать физику из одной точки.

### 4. Слишком много noisy-фичей

После генерации фичей удалить:

```python
constant columns
near-constant columns
columns with >95-99% missing, если они не являются осмысленными flags
perfect duplicates
```

### 5. Оптимизация не той метрики

Обычный AUC может расти, а `max_fpr=0.1` падать. Все решения принимать по partial AUC.

---

## Итоговая идея

Это не задача «выучить классификатор на 1000 строках». Это задача извлечь поведенческую структуру из 300k неразмеченных/тестовых следов, а 1000 labels использовать как компас.

Самые перспективные источники прироста:

```text
1. impossible/consistency flags
2. jerk + curvature + minimum-jerk residual
3. symbolic entropy + transition entropy
4. template clusters / near-duplicate mining
5. SSL trajectory embeddings + reconstruction error
```
