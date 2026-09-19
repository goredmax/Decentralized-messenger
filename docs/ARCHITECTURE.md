# Архитектура децентрализованного мессенджера

## Обзор

Этот проект реализует приватный децентрализованный мессенджер с акцентом на:
1. **E2EE (End-to-End Encryption)** - сквозное шифрование
2. **Децентрализацию** - P2P сеть без центральных серверов
3. **Tor интеграцию** - анонимизация трафика
4. **Минимизацию метаданных** - защита информации о коммуникации

## Криптографическое ядро

### Используемые алгоритмы

| Компонент | Алгоритм | Библиотека | Назначение |
|-----------|----------|------------|------------|
| Key Exchange | X25519 (ECDH) | PyNaCl | Обмен ключами |
| Encryption | AES-256-GCM | cryptography | Шифрование сообщений |
| Signatures | Ed25519 | PyNaCl | Цифровые подписи |
| Key Derivation | HMAC-SHA256 | hashlib | KDF для цепочек ключей |
| Ratchet Protocol | Double Ratchet | собственная | Forward secrecy |

### Double Ratchet Protocol

Реализован в `src/crypto/double_ratchet.py`:

```
┌─────────────────────────────────────────────────────────┐
│                  Double Ratchet                         │
├─────────────────────────────────────────────────────────┤
│  DH Ratchet (асимметричный)                            │
│  • Генерация новых ephemeral ключей для каждого сообщения│
│  • X25519 DH exchange при смене направления             │
│  • Обновляет root chain                                 │
├─────────────────────────────────────────────────────────┤
│  Symmetric Ratchet (симметричный)                      │
│  • HMAC-based KDF для эволюции цепочки ключей          │
│  • Быстрое обновление для каждого сообщения            │
│  • Message keys удаляются после использования          │
└─────────────────────────────────────────────────────────┘
```

### Свойства безопасности

1. **Forward Secrecy**: Ключи сообщений удаляются после использования
2. **Post-Compromise Security**: DH ratchet "лечит" компрометацию ключей
3. **Out-of-Order Delivery**: Временное хранение skipped keys

## P2P Сеть

### Libp2p Stack

```
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

## Tor Интеграция

### Hidden Services v3

Каждый узел может работать как Tor hidden service:
- `.onion` адрес генерируется из публичного ключа
- Трафик анонимизирован через 3+ relay узлов
- Нет возможности определить IP собеседника

### Circuit Management

```python
# Каждый новый контакт = новый Tor circuit
# Rotating circuits для минимизации correlation attacks
```

## Минимизация метаданных

### Техники

1. **Padding**: Все сообщения дополняются до фиксированного размера
2. **Timing Obfuscation**: Случайные задержки перед отправкой
3. **Contact Discovery**: Через DHT без раскрытия социальных графов
4. **No Logs**: Никаких логов соединений или сообщений

### Формат сообщения

```
┌──────────────┬─────────────┬──────────────┬────────────┐
│   Header     │    Nonce    │  Ciphertext  │   Padding  │
│  (variable)  │  (12 bytes) │  (variable)  │ (to 1KB+)  │
└──────────────┴─────────────┴──────────────┴────────────┘
```

## Структура проекта

```
/workspace
├── src/
│   ├── crypto/           # Криптография
│   │   ├── __init__.py
│   │   ├── key_management.py   # X25519, Ed25519, Box
│   │   └── double_ratchet.py   # Signal Protocol
│   ├── p2p/              # P2P сеть
│   ├── network/          # Транспорт
│   ├── tor/              # Tor интеграция
│   └── storage/          # Локальное хранилище
├── tests/                # Тесты
├── docs/                 # Документация
├── config/               # Конфигурация
├── README.md
└── requirements.txt
```

## Этапы разработки

### ✅ Phase 1: Cryptography Core
- [x] Key management с PyNaCl
- [x] Double Ratchet Protocol
- [ ] X3DH initial handshake
- [ ] Group chat protocol (Sender Keys)

### Phase 2: P2P Network
- [ ] Libp2p integration
- [ ] DHT for peer discovery
- [ ] Message routing

### Phase 3: Tor Integration  
- [ ] Hidden service setup
- [ ] Circuit management
- [ ] .onion address resolution

### Phase 4: Storage
- [ ] SQLCipher encrypted database
- [ ] Message queue
- [ ] Key storage (encrypted at rest)

### Phase 5: Client
- [ ] CLI interface
- [ ] GUI (optional)
- [ ] Mobile apps (future)

## Безопасность

### Аудит криптомодулей

Все используемые библиотеки прошли независимый аудит:
- **PyNaCl** (libsodium): multiple audits
- **cryptography.io**: regular audits
- **Libp2p**: ongoing security review

### Threat Model

Защищаемся от:
- ✓ Пассивного наблюдения трафика
- ✓ Компрометации серверов (их нет)
- ✓ Частичной компрометации ключей
- ✓ Traffic analysis (частично)

Не защищаемся от:
- ✗ Компрометации endpoint устройств
- ✗ Global passive adversary (полная корреляция)
- ✗ Социальную инженерию

## Лицензия

MIT License - способствуем распространению приватных технологий
