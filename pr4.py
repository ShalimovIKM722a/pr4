import heapq
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

MAP_SIZE = 30
NUM_RAYS = 36
SENSOR_RANGE = 5.0
STEP_SIZE = 0.2

# Реальна карта
true_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)

obstacles = [
    (4, 4), (4, 5), (4, 6), (4, 7),
    (5, 7), (6, 7), (7, 7),

    (3, 12), (4, 12), (5, 12), (6, 12),
    (7, 12), (8, 12),

    (10, 3), (10, 4), (10, 5), (10, 6),
    (11, 6), (12, 6), (13, 6),

    (11, 15), (12, 15), (13, 15),
    (14, 15), (15, 15), (16, 15),

    (18, 5), (18, 6), (18, 7), (18, 8),
    (19, 8), (20, 8), (21, 8),

    (20, 19), (21, 19), (22, 19),
    (23, 19), (24, 19),

    (7, 23), (8, 23), (9, 23), (10, 23),
    (11, 23), (12, 23),

    (25, 11), (25, 12), (25, 13),
    (25, 14), (25, 15)
]

for r, c in obstacles:
    true_map[r, c] = 1


# Початкова та кінцева позиції
start = (1, 1)
goal = (28, 28)


# A* пошук маршруту
def astar(grid, start, goal):

    def heuristic(a, b):
        return np.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2
        )

    open_set = [
        (heuristic(start, goal), 0, start, [])
    ]

    visited = set()

    while open_set:

        f, g, current, path = heapq.heappop(open_set)

        if current in visited:
            continue

        visited.add(current)
        path = path + [current]

        if current == goal:
            return path

        for dr, dc in [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]:

            nr = current[0] + dr
            nc = current[1] + dc

            if not (
                0 <= nr < MAP_SIZE and
                0 <= nc < MAP_SIZE
            ):
                continue

            if grid[nr, nc] == 1:
                continue

            if (nr, nc) in visited:
                continue

            cost = 1.414 if dr != 0 and dc != 0 else 1.0

            ng = g + cost

            heapq.heappush(
                open_set,
                (
                    ng + heuristic((nr, nc), goal),
                    ng,
                    (nr, nc),
                    path
                )
            )

    return []


# Побудова маршруту
path = astar(true_map, start, goal)

if not path:
    print("Шлях не знайдено")
    raise SystemExit

path = np.array(path, dtype=float)

# Переведення координат у формат x, y
path_xy = path[:, [1, 0]]


# Згладжування маршруту
def smooth_path(path, factor=5):

    result = []

    for i in range(len(path) - 1):

        p1 = path[i]
        p2 = path[i + 1]

        for j in range(factor):

            t = j / factor

            point = p1 * (1 - t) + p2 * t

            result.append(point)

    result.append(path[-1])

    return np.array(result)


smooth_route = smooth_path(path_xy, 5)


# Карта SLAM
slam_map = np.full(
    (MAP_SIZE, MAP_SIZE),
    -1,
    dtype=int
)


# Позиція робота
robot_true = np.array([
    smooth_route[0, 0],
    smooth_route[0, 1],
    0.0
])

robot_est = robot_true.copy()

trajectory_true = [robot_true[:2].copy()]
trajectory_est = [robot_est[:2].copy()]


# Датчик відстані
def raycast(x, y, angle):

    distance = 0.0
    step = 0.1

    while distance < SENSOR_RANGE:

        px = x + distance * np.cos(angle)
        py = y + distance * np.sin(angle)

        col = int(round(px))
        row = int(round(py))

        if not (
            0 <= row < MAP_SIZE and
            0 <= col < MAP_SIZE
        ):
            return min(distance, SENSOR_RANGE)

        if true_map[row, col] == 1:
            return distance

        distance += step

    return SENSOR_RANGE


# Оновлення карти
def update_map(pose):

    x, y, theta = pose

    for i in range(NUM_RAYS):

        angle = theta + 2 * np.pi * i / NUM_RAYS

        distance = raycast(x, y, angle)

        samples = max(1, int(distance / 0.15))

        # Вільний простір
        for j in range(samples):

            d = j * distance / samples

            px = x + d * np.cos(angle)
            py = y + d * np.sin(angle)

            col = int(round(px))
            row = int(round(py))

            if (
                0 <= row < MAP_SIZE and
                0 <= col < MAP_SIZE
            ):

                if slam_map[row, col] == -1:
                    slam_map[row, col] = 0

        # Перешкода
        if distance < SENSOR_RANGE:

            end_x = x + distance * np.cos(angle)
            end_y = y + distance * np.sin(angle)

            col = int(round(end_x))
            row = int(round(end_y))

            if (
                0 <= row < MAP_SIZE and
                0 <= col < MAP_SIZE
            ):
                slam_map[row, col] = 1


# Рух робота
for i in range(1, len(smooth_route)):

    previous = smooth_route[i - 1]
    current = smooth_route[i]

    dx = current[0] - previous[0]
    dy = current[1] - previous[1]

    distance = np.sqrt(dx ** 2 + dy ** 2)

    angle = np.arctan2(dy, dx)

    # Реальна позиція
    robot_true[0] = current[0]
    robot_true[1] = current[1]
    robot_true[2] = angle

    # Одометрія з шумом
    noisy_distance = distance + np.random.normal(
        0, 0.003
    )

    noisy_angle = angle + np.random.normal(
        0, 0.002
    )

    robot_est[0] += noisy_distance * np.cos(
        noisy_angle
    )

    robot_est[1] += noisy_distance * np.sin(
        noisy_angle
    )

    robot_est[2] = noisy_angle

    # Побудова карти
    update_map(robot_true)

    trajectory_true.append(
        robot_true[:2].copy()
    )

    trajectory_est.append(
        robot_est[:2].copy()
    )


trajectory_true = np.array(trajectory_true)
trajectory_est = np.array(trajectory_est)


# Оцінка похибки локалізації
errors = np.linalg.norm(
    trajectory_true - trajectory_est,
    axis=1
)

mean_error = errors.mean()
max_error = errors.max()
final_error = errors[-1]


# Оцінка карти
known_mask = slam_map != -1

known_cells = np.sum(known_mask)

coverage = (
    known_cells / (MAP_SIZE * MAP_SIZE)
) * 100


# Оцінка перешкод
true_obstacles = true_map[known_mask] == 1
pred_obstacles = slam_map[known_mask] == 1

TP = np.sum(true_obstacles & pred_obstacles)
FP = np.sum(~true_obstacles & pred_obstacles)
FN = np.sum(true_obstacles & ~pred_obstacles)

precision = (
    TP / (TP + FP)
    if TP + FP > 0 else 0
)

recall = (
    TP / (TP + FN)
    if TP + FN > 0 else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0 else 0
)


# Результати
print("\n========== РЕЗУЛЬТАТИ SLAM ==========")

print(f"Довжина маршруту: {len(smooth_route)}")

print(f"Середня похибка: {mean_error:.4f} м")

print(f"Максимальна похибка: {max_error:.4f} м")

print(f"Кінцева похибка: {final_error:.4f} м")

print(f"Досліджено карти: {coverage:.2f}%")

print(f"Precision: {precision:.2%}")

print(f"Recall: {recall:.2%}")

print(f"F1-score: {f1:.4f}")

print(f"Виявлено перешкод: {TP}")

print("====================================")


# Візуалізація
fig, axes = plt.subplots(
    1, 3, figsize=(18, 6)
)


# Реальна карта
axes[0].imshow(
    true_map,
    origin="lower",
    cmap="Greys",
    vmin=0,
    vmax=1
)

axes[0].plot(
    trajectory_true[:, 0],
    trajectory_true[:, 1],
    label="Реальна траєкторія"
)

axes[0].scatter(
    start[1], start[0],
    marker="o",
    label="Старт"
)

axes[0].scatter(
    goal[1], goal[0],
    marker="x",
    label="Ціль"
)

axes[0].set_title("Реальна карта")
axes[0].legend()


# Карта SLAM
shown_map = np.ma.masked_where(
    slam_map == -1,
    slam_map
)

axes[1].imshow(
    shown_map,
    origin="lower",
    cmap="viridis",
    vmin=0,
    vmax=1
)

axes[1].plot(
    trajectory_true[:, 0],
    trajectory_true[:, 1],
    label="Траєкторія"
)

axes[1].set_title("Карта SLAM")
axes[1].legend()


# Порівняння траєкторій
axes[2].plot(
    trajectory_true[:, 0],
    trajectory_true[:, 1],
    label="Реальна траєкторія"
)

axes[2].plot(
    trajectory_est[:, 0],
    trajectory_est[:, 1],
    "--",
    label="Оцінка одометрії"
)

axes[2].set_title("Порівняння траєкторій")
axes[2].legend()


for ax in axes:

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_xlim(0, MAP_SIZE - 1)
    ax.set_ylim(0, MAP_SIZE - 1)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()