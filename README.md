# AntiTerror — интеграция с фронтендом

Этот проект предоставляет Python‑слой интеграции, который позволяет фронтенду
получать несколько потоков камер с JPEG‑кадрами и живой статистикой.

## Быстрый старт (несколько камер)

```python
from anti_terror.service import MultiCameraService, CameraConfig

svc = MultiCameraService()

# Опционально: получать события/алерты
def on_events(camera_id, events):
    print("events", camera_id, events)

svc.set_event_callback(on_events)

# Запуск камер
svc.start_camera(CameraConfig(camera_id="CAM_01", source=0, db_path="antiterror.db"))
svc.start_camera(CameraConfig(camera_id="CAM_02", source="rtsp://user:pass@host/stream"))

# Получить последний JPEG/статистику (для передачи во фронтенд)
jpeg = svc.get_latest_jpeg("CAM_01")
stats = svc.get_latest_stats("CAM_01")

# Остановить камеры
svc.stop_camera("CAM_01")
svc.stop_all()
```

## API интеграции

### MultiCameraService

- `start_camera(cfg: CameraConfig)`: запускает пайплайн в фоне.
- `stop_camera(camera_id: str)`: останавливает одну камеру.
- `stop_all()`: останавливает все камеры.
- `get_latest_jpeg(camera_id: str) -> bytes | None`: последний кадр в JPEG.
- `get_latest_stats(camera_id: str) -> dict | None`: последняя статистика.
- `set_event_callback(callback)`: колбек получает `(camera_id, events)`.

### CameraConfig

```python
CameraConfig(
    camera_id: str,        # unique camera identifier
    source: str | int,     # webcam index or path/RTSP URL
    db_path: str = "antiterror.db",
    device: str = "cuda",  # "cuda" | "mps" | "cpu"
    preview_port: int | None = None,
    render_enabled: bool = False,
)
```

Примечания:
- `preview_port` запускает простой MJPEG‑сервер (для быстрых проверок).
- `render_enabled` включает OpenCV‑окно (полезно только локально).

## Форматы данных

### Статистика

```python
{
  "camera_id": "CAM_01",
  "faces": 3,
  "ids": 2,
  "bags": 1,
  "owned": 1,
  "session_id": "S_2026_02_03_0001"
}
```

### События

`events` — список словарей, формируемых анализатором поведения; каждый элемент
минимум содержит `type`, а также может включать `bag_id`, `person_id` и другие
поля в зависимости от типа события.

## Рекомендации по отдаче во фронтенд

Два распространённых варианта интеграции:

1) HTTP API:
   - Сделать небольшой API, который отдаёт `get_latest_jpeg` и `get_latest_stats`.
   - Фронтенд опрашивает API и рисует кадры как изображения.

2) WebSocket/Stream:
   - Пушить JPEG и статистику по WS для меньшей задержки.

Если нужно, добавлю FastAPI‑сервер с эндпоинтами:
- `/cameras` список
- `/camera/{id}/jpeg`
- `/camera/{id}/stats`
- `/events` поток

## Диагностика

- Если камера не открывается, проверьте корректность `source`.
- Если кадров нет, проверьте доступ к GPU или переключитесь на `device="cpu"`.
- RTSP‑потоки часто требуют корректных логина/пароля и сетевого доступа.
