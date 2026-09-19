# Децентрализованный мессенджер с E2EE и защитой метаданных

**Статус:** Reference Implementation (Python) → Production (Rust)

> ⚠️ **Важно:** Это reference implementation для прототипирования и аудита протокола. Для production использования планируется переписывание ядра на Rust (snow, libsignal-client, arti для Tor).

## Цель
Создание приватного децентрализованного мессенджера с:
- **End-to-End Encryption (E2EE)** - сквозное шифрование на основе X3DH + Double Ratchet (Signal Protocol)
- **Децентрализация** - P2P архитектура без центральных серверов
- **Tor интеграция** - анонимизация трафика через Tor Onion Services v3
- **Минимизация метаданных** - padding, timing obfuscation, cover traffic

## Архитектура

### Компоненты

1. **Криптография (`src/crypto/`)**
   - X25519 для обмена ключами (ECDH)
   - AES-256-GCM для шифрования сообщений
   - Ed25519 для цифровых подписей
   - Double Ratchet Protocol (как в Signal Protocol)

2. **P2P сеть (`src/p2p/`)**
   - Libp2p для децентрализованной сети
   - DHT (Distributed Hash Table) для обнаружения узлов
   - Kademlia для маршрутизации

3. **Tor интеграция (`src/tor/`)**
   - Onion routing для анонимизации
   - Hidden Services (.onion адреса)
   - Circuit management

4. **Сеть (`src/network/`)**
   - QUIC протокол для транспорта
   - Multiplexing соединений
   - NAT traversal

5. **Хранилище (`src/storage/`)**
   - Локальное зашифрованное хранилище
   - SQLite с SQLCipher
   - Ephemeral messages support

## Используемые технологии

### Криптография
- **libsodium** - проверенная криптографическая библиотека
- **noise-protocol** - Noise Protocol Framework

### P2P
- **libp2p** - модульная сетевая стек
- **ipfs** - распределённое хранение (опционально)

### Tor
- **tor** - основной Tor демон
- **stem** - Python контроллер для Tor
- **pytor** - интеграция с Python

### Транспорт
- **QUIC** - современный транспортный протокол
- **WebRTC** - для browser клиентов

## Структура проекта

```
/workspace
├── src/
│   ├── crypto/          # Криптографические примитивы
│   ├── p2p/             # P2P сеть и DHT
│   ├── network/         # Сетевой транспорт
│   ├── tor/             # Tor интеграция
│   └── storage/         # Локальное хранилище
├── tests/               # Тесты
├── docs/                # Документация
├── config/              # Конфигурационные файлы
└── README.md
```

## Безопасность

### E2EE реализация
1. Генерация ключей на устройстве пользователя
2. Обмен ключами через X25519 ECDH
3. Double Ratchet для forward secrecy
4. Подпись сообщений через Ed25519

### Защита метаданных
1. Использование Tor скрытых сервисов
2. Padding сообщений до фиксированного размера
3. Периодическая ротация идентификаторов
4. No persistent connection logs

### Децентрализация
1. Отсутствие центральных серверов
2. DHT для discovery peers
3. Gossip protocol для распространения сообщений
4. Mesh network topology

## Этапы разработки

### Phase 1: Cryptography Core
- [ ] Реализация генерации ключей
- [ ] E2EE шифрование/дешифрование
- [ ] Double Ratchet Protocol

### Phase 2: P2P Network
- [ ] Интеграция libp2p
- [ ] DHT implementation
- [ ] Peer discovery

### Phase 3: Tor Integration
- [ ] Tor hidden services
- [ ] Circuit management
- [ ] Onion routing

### Phase 4: Messaging Protocol
- [ ] Message format
- [ ] Delivery guarantees
- [ ] Group chats

### Phase 5: Storage & UI
- [ ] Encrypted local storage
- [ ] CLI client
- [ ] GUI client (optional)

## Требования к безопасности

1. **Никогда не хранить**:
   - Приватные ключи на сервере
   - Логи сообщений
   - Метаданные о контактах

2. **Всегда использовать**:
   - Проверенные криптобиблиотеки
   - Forward secrecy
   - Perfect forward secrecy (PFS)

3. **Защищать от**:
   - Timing attacks
   - Traffic analysis
   - Sybil attacks

## Лицензия
MIT License

## Contributing
Приветствуются contributions focused на security и privacy!
