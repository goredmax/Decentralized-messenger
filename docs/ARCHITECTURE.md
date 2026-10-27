# Архитектура: текущее состояние и цель

**Статус на 2026-10-26.**

Документ отвечает на два разных вопроса, и раньше они были смешаны:

1. что реально реализовано в репозитории и проверено тестами;
2. что задумано, но кода ещё нет.

Границы гарантий и известные дыры перечислены в
[docs/SECURITY_BOUNDARIES.md](SECURITY_BOUNDARIES.md), модель угроз — в
[THREAT_MODEL.md](../THREAT_MODEL.md). Этот документ их не дублирует.

---

## 1. Текущее состояние

### 1.1. Как собирается сессия

```text
PreKeyBundle, подписанный Ed25519
        │
        ▼
X3DH / PQXDH: X25519 + ML-KEM-768
        │
        ▼
root key ── HKDF-SHA256 ──► Double Ratchet
        │
        ▼
key confirmation (MAC) ──► сообщения под XChaCha20-Poly1305
```

Что здесь важно:

- подпись prekey-bundle проверяется до того, как будет потрачен одноразовый
  prekey;
- handshake инициатора подписан Ed25519 и покрывает transcript, включая
  ML-KEM-768 ciphertext;
- root key на стороне инициатора заперт до проверки key confirmation MAC;
- Double Ratchet привязан к сессии через associated data, выведенные из обоих
  identity key и версии протокола.

### 1.2. Криптография

| Компонент | Реализация | Библиотека |
| --- | --- | --- |
| Обмен ключами | X25519 ECDH + ML-KEM-768 (PQXDH) | PyNaCl, `cryptography` |
| Подписи и identity | Ed25519 | PyNaCl |
| Шифрование сообщений | XChaCha20-Poly1305 | PyNaCl (libsodium) |
| Корневой KDF | HKDF-SHA256 (RFC 5869) | собственная реализация в `src/crypto/kdf.py`, сверяется с `cryptography` и векторами RFC |
| Цепочки Double Ratchet | HMAC-SHA256 | `hmac`/`hashlib` |
| Хранилище | scrypt → HKDF-SHA256 → XChaCha20-Poly1305 | `hashlib`, PyNaCl |

`cryptography` нужна для ML-KEM-768 и для кросс-валидации примитивов в тестах;
шифрование сообщений и хранилища идёт через PyNaCl.

Отступление от Signal, зафиксированное явно: identity key — это Ed25519, и он
же конвертируется в X25519 для Diffie-Hellman. В Signal для подписи и для DH
используются разные ключи. Компрометация одного identity key здесь ломает и
аутентификацию, и key agreement.

### 1.3. Протокол и wire-формат

- `src/crypto/x3dh.py` — X3DH, PQXDH, проверка подписи bundle и handshake,
  key confirmation, резерв/расход одноразовых prekey.
- `src/crypto/double_ratchet.py` — DH-ratchet и симметричный ratchet,
  skipped message keys, сериализация состояния, associated data.
- `src/protocol/session.py` — связывает X3DH с Double Ratchet; инициатор не
  может отправить сообщение до подтверждения ключа.
- `src/protocol/wire.py` — канонический формат кадра версии 3:

  ```text
  +---------+---------+------------------+-------------------+
  | version | kind    | payload_len      | payload           |
  | 1 byte  | 1 byte  | varint (1..9 B)  | payload_len bytes |
  +---------+---------+------------------+-------------------+
  ```

  Для ratchet-сообщения payload состоит из `dh(32)`, varint `pn`, varint `n`,
  varint длины ciphertext, самого ciphertext (nonce 24 байта + tag 16 байт) и
  необязательного заполнения до границы bucket.

- `src/protocol/padding.py` — два слоя сокрытия длины: паддинг plaintext под
  AEAD и квантование длины кадра. Профиль по умолчанию: floor 8192 байта,
  шаг 8192. Наблюдателю всё равно видна граница bucket, число сообщений и
  тайминги; это записано в модуле и в границах безопасности.
- `src/protocol/sender_key.py` — Sender Keys для групп: chain key, производные
  message keys, distribution message, ротация после удаления участника.
  Распространение distribution-сообщений между процессами не реализовано:
  транспорта для этого нет.
- `src/protocol/transparency.py` — подписанный hash-linked лог цепочек Sender
  Keys. Ловит повтор, форк, откат и дыру; **не** является key transparency,
  потому что независимых свидетелей и gossip нет.
- `src/protocol/verifier.py` — safety numbers для внеполосной проверки
  identity key.

Декодеры `wire.py` тотальные: на любом входе они обязаны вернуть ошибку
`WireError`, а не `KeyError`/`struct.error`, и проверяют объявленную длину до
выделения буфера.

### 1.4. Хранилище

`src/storage/container.py` — зашифрованный контейнер:

```text
header  = magic(8) | version(1) | kdf_id(1) | N_log2(1) | r(1) | p(1) | salt(16)
body    = nonce(24) | ciphertext
payload = epoch(8) | plaintext
```

Ключ выводится из пароля через scrypt, затем через HKDF-SHA256. Заголовок
передаётся в AEAD как associated data, поэтому параметры KDF тоже
аутентифицированы. Epoch лежит внутри зашифрованного payload и позволяет
отказаться от более старой версии файла, если у вызывающего есть внешний
anchor. Замена файла целиком на более старый, но внутренне валидный **не
обнаруживается**.

Поверх контейнера работают `PersistentPreKeyStore`,
`PersistentPQPreKeyStore`, `IdentityStore`, `SessionStore` и `PeerDirectory`.
Это не SQLite и не SQLCipher: отдельной базы данных в проекте нет.

### 1.5. Транспорт

`src/p2p/relay.py` определяет интерфейс `Transport` и реализует relay в одном
процессе: две стороны обмениваются кадрами через очередь, relay записывает
длины кадров. Очередь ограничена, чтобы медленный получатель не приводил к
неограниченному росту памяти.

Это не сеть: нет сокетов, портов, discovery и соединений между машинами.
`src/tor/` — пустая заглушка. `src/network/` не существует.

### 1.6. Что лежит в репозитории

```text
src/
├── crypto/       x3dh.py, double_ratchet.py, kdf.py, key_management.py,
│                 prekey_store.py, pq.py
├── protocol/     session.py, wire.py, padding.py, sender_key.py,
│                 transparency.py, verifier.py, limits.py
├── p2p/          relay.py
├── storage/      container.py, store.py
└── tor/          пустая заглушка
examples/         cli.py — демонстрация полного цикла в одном процессе
tests/            685 тестов
docs/             ARCHITECTURE.md, SECURITY_BOUNDARIES.md
```

В корне: `pyproject.toml`, `pytest.ini`, `requirements*.in`,
`requirements*.lock`. Файлов `requirements.txt`, каталога `config/` и
каталога `src/network/` нет.

`examples/cli.py` — не мессенджер, а проверка того, что протокол сходится:
bundle → X3DH/PQXDH → key confirmation → Double Ratchet → сообщения в обе
стороны через relay. Он показывает safety numbers и длины кадров, то есть
ровно то, что видит пассивный наблюдатель.

---

## 2. Целевая архитектура (не реализовано)

Всё в этом разделе — план, а не описание текущего кода. Ни один из компонентов
ниже не входит в зависимости и не проверяется тестами.

### 2.1. Сеть

```text
┌─────────────────────────────────────────┐
│           Application Layer             │
│         (Messaging Protocol)            │
├─────────────────────────────────────────┤
│           Secure Channel                │
│         (Noise Protocol)                │
├─────────────────────────────────────────┤
│          Transport Layer                │
│     (QUIC / TCP / WebRTC)               │
├─────────────────────────────────────────┤
│        Peer Discovery Layer             │
│      (Kademlia DHT / mDNS)              │
├─────────────────────────────────────────┤
│          Network Layer                  │
│       (IP / Tor Onion)                  │
└─────────────────────────────────────────┘
```

libp2p, DHT, QUIC, Noise и NAT traversal не реализованы. libp2p нет в
зависимостях, и утверждения о его аудите к этому проекту не относятся.

### 2.2. Tor

Onion services v3, управление цепочками и ротация circuit'ов не реализованы.
Каталог `src/tor/` существует только как место для будущего кода.

### 2.3. Метаданные

Реализовано только сокрытие длины сообщения ступенчатым паддингом. Не
реализованы:

- cover traffic;
- обфускация таймингов и случайные задержки;
- межкадровый padding как отдельный слой;
- contact discovery через DHT без раскрытия социального графа.

В коде нет логирования сообщений и соединений, но это свойство текущего кода,
а не гарантия для окружения, в котором он запущен.

### 2.4. Клиент

`examples/cli.py` — демонстрационный CLI. Полноценного клиента, GUI и мобильных
приложений нет.

### 2.5. Этапы

- **Phase 1: криптографическое ядро**
  - [x] key management и конвертация Ed25519 → X25519
  - [x] Double Ratchet
  - [x] X3DH
  - [x] PQXDH с ML-KEM-768
  - [x] key confirmation
  - [x] Sender Keys (шифрование и distribution message; доставки нет)
- **Phase 2: P2P-сеть**
  - [ ] libp2p
  - [ ] DHT для discovery
  - [ ] маршрутизация сообщений
  - [x] интерфейс `Transport` и relay в одном процессе (как заготовка)
- **Phase 3: Tor**
  - [ ] hidden service
  - [ ] управление цепочками
  - [ ] разрешение `.onion`
- **Phase 4: хранилище**
  - [x] шифрованный контейнер (scrypt + XChaCha20-Poly1305)
  - [x] хранение ключей и состояния сессий
  - [ ] очередь сообщений
  - [ ] SQLCipher или другая SQL-база (не используется)
- **Phase 5: клиент**
  - [x] демонстрационный CLI
  - [ ] полноценный CLI-клиент
  - [ ] GUI
  - [ ] мобильные приложения

---

## 3. Безопасность

Проект **не проходил независимый аудит**. Это прототип криптографического
ядра, а не готовый мессенджер.

PyNaCl/libsodium и `cryptography` имеют собственные аудиты и репутацию, но это
аудиты библиотек, а не этого кода и не его композиции. Упоминание аудита
libp2p из предыдущей версии документа удалено: libp2p здесь не используется.

Подробная модель угроз — в [THREAT_MODEL.md](../THREAT_MODEL.md). Точные границы
того, что защищено, что не защищено и какие ограничения невозможно закрыть
кодом в этом репозитории, — в
[docs/SECURITY_BOUNDARIES.md](SECURITY_BOUNDARIES.md).

## 4. Лицензия

MIT License.
