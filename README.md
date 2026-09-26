# Децентрализованный мессенджер с E2EE и защитой метаданных

> ⚠️ **Не используйте это для реальных секретов.**
> Это неаудированный прототип криптографического ядра. Сеть, Tor, хранилище и
> транспорт **не реализованы**. Независимого аудита не было. Автор этих правок не
> является аудитором: исправления написаны тем же кодом, который их проверяет.

**Статус:** Reference implementation (Python) → Production (Rust)

## Цель

- **End-to-End Encryption (E2EE)** — X3DH + Double Ratchet (Signal Protocol)
- **Децентрализация** — P2P без центральных серверов *(не реализовано)*
- **Tor** — анонимизация трафика через onion services *(не реализовано)*
- **Минимизация метаданных** — padding, timing obfuscation, cover traffic
  *(не реализовано)*

## Что реально есть в репозитории

| Модуль | Состояние |
| --- | --- |
| `src/crypto/x3dh.py` | Реализован. Аутентификация prekey-bundle и handshake. |
| `src/crypto/double_ratchet.py` | Реализован. Соответствует спецификации Signal. |
| `src/crypto/prekey_store.py` | Реализован. Атомарный одноразовый учёт prekey. |
| `src/crypto/kdf.py` | Реализован. HKDF-SHA256 (RFC 5869). |
| `src/crypto/key_management.py` | Реализован. Обёртки над PyNaCl. |
| `src/p2p/`, `src/tor/`, `src/storage/` | **Пустые заглушки.** |
| `src/network/` | Не существует. |

Документация в `docs/` и `THREAT_MODEL.md` описывает целевую архитектуру, а не
текущее состояние кода.

## Архитектура (целевая)

1. **Криптография (`src/crypto/`)** — X25519 (ECDH), Ed25519 (подписи),
   XChaCha20-Poly1305 (AEAD в ратчете)
2. **P2P сеть (`src/p2p/`)** — libp2p, DHT, Kademlia
3. **Tor (`src/tor/`)** — onion services v3, управление цепочками
4. **Сеть (`src/network/`)** — QUIC, мультиплексирование, NAT traversal
5. **Хранилище (`src/storage/`)** — SQLite + SQLCipher, ephemeral messages

## Используемые технологии

- **PyNaCl / libsodium** — примитивы, единственная runtime-зависимость
- **cryptography** — только для тестов (кросс-валидация против независимой
  реализации)

### Зависимости

| Файл | Назначение |
| --- | --- |
| `requirements.in` | Runtime: только то, что реально импортирует `src/` |
| `requirements.lock` | Hashed lock для `pip install --require-hashes` |
| `requirements-dev.in` | Тесты и статический анализ |
| `requirements-dev.lock` | Hashed lock для CI |

Раньше в `requirements.txt` были `libp2p`, `stem`, `tor-request`, `aiohttp`,
`aiortc`, `sqlcipher-legacy` и `orjson` с диапазонами `>=` без хешей. Ни один из
них не импортировался: соответствующие модули — пустые заглушки.
`sqlcipher-legacy` при этом не является рабочим биндингом SQLCipher (нужен
`pysqlcipher3`). Теперь зависимости совпадают с кодом, а lock-файлы позволяют
воспроизводимую установку.

Перегенерировать:

```sh
pip-compile --generate-hashes --output-file=requirements.lock requirements.in
pip-compile --generate-hashes --allow-unsafe \
    --output-file=requirements-dev.lock requirements-dev.in
```

> Lock-файлы собраны под Python 3.13. Расширение матрицы версий в CI требует
> пересборки lock-файлов.

## Безопасность

### E2EE

1. Ключи идентичности генерируются на устройстве (Ed25519)
2. **Prekey-bundle аутентифицируется**: подпись signed prekey проверяется
   инициатором до использования (`X3DH.verify_bundle`)
3. **Handshake аутентифицируется**: инициатор подписывает транскрипт, получатель
   проверяет подпись и берёт identity key из самого транскрипта, а не от
   вызывающего кода
4. **One-time prekey строго одноразовые** — атомарный `consume` в
   `PreKeyStore`; повторное использование прерывает handshake
5. Double Ratchet даёт forward secrecy и post-compromise security
6. **Сессия восстанавливается только через `DoubleRatchet.from_state`**,
   который перепроверяет всё состояние, включая соответствие
   `dh_local_pub` приватному ключу. Повреждённый state-файл приводит к
   явной ошибке, а не к сессии, которая шифрует «вроде бы успешно», а
   собеседник расшифровать не может

### Отклонения от спецификации Signal

Осознанные и задокументированные:

- **HKDF `info` = непустой context string.** Спека оставляет `info` пустым.
  Непустой контекст — это domain separation, он не ослабляет конструкцию, но
  делает ключи **несовместимыми с libsignal**.
- **Ed25519 → X25519 конверсия identity key.** Спека использует отдельные
  ключи для подписи и для DH. Здесь один ключ играет обе роли, что означает
  компрометацию обеих сразу. Конверсия собрана в одном месте
  (`key_management.identity_*_x25519`).
- **Версия протокола 2.** `KDF_CK` в исходном коде был перевёрнут относительно
  спеки (`ck = HMAC(0x01)`, `mk = HMAC(0x02)`), а строка HKDF была другой.
  Сессии предыдущей версии **отвергаются при загрузке**, а не интерпретируются
  наугад.

### Известные ограничения

- Независимого аудита нет
- **Официальных тест-вектор Signal для X3DH и Double Ratchet не существует.**
  Signal не публикует их: ни в [X3DH rev 1](https://signal.org/docs/specifications/x3dh/),
  ни в [Double Ratchet rev 4](https://signal.org/docs/specifications/doubleratchet/)
  нет приложения с тест-векторами — оба документа заканчиваются разделом
  References. Поэтому составные протоколы проверяются только кросс-валидацией
  против `cryptography` и round-trip тестами
- Примитивы же проверены по **опубликованным векторам RFC**: HKDF-SHA256
  (RFC 5869, включая промежуточный PRK), X25519 (RFC 7748), Ed25519 (RFC 8032) —
  см. `tests/test_vectors.py`. Это не замена аудиту, но это внешняя точка
  отсчёта, а не согласие двух реализаций
- Защита от глобальной корреляции трафика требует mixnet, её нет
- Тайминг-анализ на уровне приложения не закрыт полностью

### Независимое подтверждение

Не прислано и не заявлено.

## Тесты

```sh
pip install --require-hashes -r requirements-dev.lock
python -m pytest tests/ -v
```

| Файл | Что проверяет |
| --- | --- |
| `tests/test_x3dh_mitm.py` | Регрессии аутентификации: подмена signed prekey, подмена ephemeral key, чужой identity key, replay one-time prekey |
| `tests/test_x3dh.py` | Согласование ключей, KDF, отклонение low-order точек |
| `tests/test_prekey_store.py` | Одноразовость prekey, атомарность под конкуренцией |
| `tests/test_double_ratchet.py` | Порядок `KDF_CK`, out-of-order, откат состояния, валидация заголовка, границы skipped keys |
| `tests/test_from_state.py` | Восстановление сессии: продолжение диалога после перезагрузки, отказ на повреждённом состоянии, согласованность `dh_local_pub`/`dh_local_priv` |
| `tests/test_primitives.py` | Кросс-валидация HKDF и X25519 против `cryptography` |
| `tests/test_vectors.py` | Опубликованные вектора RFC 5869 (HKDF), RFC 7748 (X25519), RFC 8032 (Ed25519) + закрепление констант протокола |

## Лицензия

MIT, см. [LICENSE](LICENSE).

## Безопасность

**Точная граница возможностей — в [docs/SECURITY_BOUNDARIES.md](docs/SECURITY_BOUNDARIES.md)**:
что защищено, что нет, карта констант-тайма, статус обнуления памяти и что
требует внешнего аудита или Rust-ядра. Этот документ стоит прочитать до
использования, а не после.

Политика раскрытия и список того, что считается уязвимостью, — в
[SECURITY.md](SECURITY.md). Модель угроз с указанием того, что реально
реализовано, — в [THREAT_MODEL.md](THREAT_MODEL.md).

## Contributing

Приветствуются contributions, связанные с security и privacy, особенно:
официальные тест-вектора Signal, независимый аудит, реализация хранилища.
