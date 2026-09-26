"""
Tests for the canonical wire encoding.

The decoders are the boundary where untrusted bytes enter the process, so the
tests concentrate on malformation rather than happy paths. Two properties matter
most:

1. Every decoder is total. On any input it raises WireError, never KeyError,
   struct.error or an IndexError, and never loops without bound.
2. One logical frame has exactly one encoding. A non-minimal varint is rejected,
   so an attacker cannot produce two distinct byte strings that decode to the
   same message.
"""

import random

import pytest
from nacl.public import PrivateKey

from src.crypto.prekey_store import InMemoryPreKeyStore
from src.crypto.x3dh import (
    X3DH,
    X3DHResponder,
    InvalidSignatureError,
)
from src.protocol.wire import (
    KIND_HANDSHAKE_INIT,
    KIND_PREKEY_BUNDLE,
    KIND_RATCHET_MESSAGE,
    KEY_LEN,
    MAX_FRAME_BYTES,
    WIRE_VERSION,
    FrameTooLarge,
    InvalidField,
    InvalidVarint,
    TrailingData,
    TruncatedFrame,
    UnknownMessageKind,
    UnsupportedVersion,
    WireError,
    decode_handshake_init,
    decode_prekey_bundle,
    decode_ratchet_message,
    decode_varint,
    encode_handshake_init,
    encode_prekey_bundle,
    encode_ratchet_message,
    encode_varint,
)


def payload_offset(wire: bytes) -> int:
    """
    Byte offset of the payload inside a frame.

    Tests must not hardcode this: the header is ``version | kind | varint`` and
    the varint takes two bytes once the payload exceeds 127, which it always
    does here. Getting this wrong silently mutates the wrong field.
    """
    length, offset = decode_varint(wire, 2)
    assert offset + length == len(wire)
    return offset


def make_bundle(with_one_time_prekey=True):
    x3dh = X3DH()
    identity_private, _ = x3dh.generate_identity_keys()
    spk_private, _, _ = x3dh.generate_signed_prekey(identity_private)
    store = InMemoryPreKeyStore()
    if with_one_time_prekey:
        store.put(x3dh.generate_one_time_prekey(key_id=4242))
    responder = X3DHResponder(
        identity_private=identity_private,
        signed_prekey_private=spk_private,
        prekey_store=store,
    )
    return x3dh, responder


def make_init(with_one_time_prekey=True):
    x3dh, responder = make_bundle(with_one_time_prekey)
    alice_private, _ = x3dh.generate_identity_keys()
    _, init = x3dh.initiate_handshake(alice_private, responder.publish_bundle())
    return x3dh, responder, init


class TestVarint:
    @pytest.mark.parametrize('value', [0, 1, 127, 128, 300, 2 ** 16, 2 ** 32, 2 ** 63])
    def test_round_trip(self, value):
        encoded = encode_varint(value)
        decoded, offset = decode_varint(encoded, 0)
        assert decoded == value
        assert offset == len(encoded)

    def test_encoding_is_minimal(self):
        assert encode_varint(0) == b'\x00'
        assert encode_varint(1) == b'\x01'
        assert encode_varint(127) == b'\x7f'
        assert encode_varint(128) == b'\x80\x01'
        assert encode_varint(300) == b'\xac\x02'

    def test_non_canonical_rejected(self):
        # 0 encodes as one byte; two bytes is the same value, so it is refused.
        with pytest.raises(InvalidVarint):
            decode_varint(b'\x80\x00', 0)
        with pytest.raises(InvalidVarint):
            decode_varint(b'\x81\x80\x00', 0)

    def test_oversized_rejected(self):
        with pytest.raises(InvalidVarint):
            decode_varint(b'\xff' * 12, 0)

    def test_truncated_rejected(self):
        with pytest.raises(TruncatedFrame):
            decode_varint(b'\x80', 0)

    def test_offset_is_respected(self):
        encoded = encode_varint(9)
        value, offset = decode_varint(b'\xff' + encoded, 1)
        assert value == 9 and offset == 2

    @pytest.mark.parametrize('value', [-1, 2 ** 64])
    def test_out_of_range_rejected(self, value):
        with pytest.raises(ValueError):
            encode_varint(value)

    def test_bool_rejected(self):
        with pytest.raises(TypeError):
            encode_varint(True)


class TestFrameHeader:
    def test_short_frame_rejected(self):
        for data in (b'', b'\x01', b'\x01\x01'):
            with pytest.raises(TruncatedFrame):
                decode_prekey_bundle(data)

    def test_unknown_version_rejected(self):
        _, responder = make_bundle()
        wire = bytearray(encode_prekey_bundle(responder.publish_bundle()))
        wire[0] = WIRE_VERSION + 1
        with pytest.raises(UnsupportedVersion):
            decode_prekey_bundle(bytes(wire))

    def test_unknown_kind_rejected(self):
        _, responder = make_bundle()
        wire = bytearray(encode_prekey_bundle(responder.publish_bundle()))
        wire[1] = 0x7F
        with pytest.raises(UnknownMessageKind):
            decode_prekey_bundle(bytes(wire))

    def test_kind_must_match_decoder(self):
        _, responder = make_bundle()
        wire = encode_prekey_bundle(responder.publish_bundle())
        with pytest.raises(UnknownMessageKind):
            decode_handshake_init(wire)

    def test_oversized_declared_length_rejected_before_allocation(self):
        # Declares a 2**40 byte payload. Must fail on the declaration, not by
        # trying to allocate it.
        header = bytes((WIRE_VERSION, KIND_PREKEY_BUNDLE)) + encode_varint(2 ** 40)
        with pytest.raises(FrameTooLarge):
            decode_prekey_bundle(header)

    def test_truncation_at_every_offset_rejected(self):
        _, responder = make_bundle()
        wire = encode_prekey_bundle(responder.publish_bundle())
        for cut in range(len(wire)):
            with pytest.raises(WireError):
                decode_prekey_bundle(wire[:cut])

    def test_trailing_data_rejected(self):
        _, responder = make_bundle()
        wire = encode_prekey_bundle(responder.publish_bundle())
        with pytest.raises(TrailingData):
            decode_prekey_bundle(wire + b'\x00')

    def test_trailing_garbage_rejected(self):
        _, responder = make_bundle()
        wire = encode_prekey_bundle(responder.publish_bundle())
        with pytest.raises(TrailingData):
            decode_prekey_bundle(wire + b'junkjunk')


class TestPreKeyBundle:
    def test_round_trip_with_one_time_prekey(self):
        _, responder = make_bundle(True)
        bundle = responder.publish_bundle()
        decoded = decode_prekey_bundle(encode_prekey_bundle(bundle))
        assert encode_prekey_bundle(decoded) == encode_prekey_bundle(bundle)
        assert decoded.one_time_prekey_id == bundle.one_time_prekey_id

    def test_round_trip_without_one_time_prekey(self):
        _, responder = make_bundle(False)
        bundle = responder.publish_bundle()
        decoded = decode_prekey_bundle(encode_prekey_bundle(bundle))
        assert decoded.one_time_prekey is None
        assert decoded.one_time_prekey_id is None

    def test_decoded_bundle_still_authenticates(self):
        _, responder = make_bundle()
        wire = encode_prekey_bundle(responder.publish_bundle())
        X3DH.verify_bundle(decode_prekey_bundle(wire))

    def test_tampered_signature_rejected_by_verification(self):
        _, responder = make_bundle()
        wire = bytearray(encode_prekey_bundle(responder.publish_bundle()))
        base = payload_offset(bytes(wire))
        wire[base + 32 + 32 + 63] ^= 0xFF   # last byte of the signature
        decoded = decode_prekey_bundle(bytes(wire))
        with pytest.raises(InvalidSignatureError):
            X3DH.verify_bundle(decoded)

    def test_bad_one_time_flag_rejected(self):
        _, responder = make_bundle(False)
        wire = bytearray(encode_prekey_bundle(responder.publish_bundle()))
        base = payload_offset(bytes(wire))
        wire[base + 32 + 32 + 64] = 0x02
        with pytest.raises(InvalidField):
            decode_prekey_bundle(bytes(wire))

    def test_flag_one_without_key_rejected(self):
        _, responder = make_bundle(False)
        wire = bytearray(encode_prekey_bundle(responder.publish_bundle()))
        base = payload_offset(bytes(wire))
        wire[base + 32 + 32 + 64] = 0x01
        with pytest.raises(TruncatedFrame):
            decode_prekey_bundle(bytes(wire))


class TestHandshakeInit:
    def test_round_trip(self):
        _, _, init = make_init(True)
        wire = encode_handshake_init(init)
        decoded = decode_handshake_init(wire)
        assert encode_handshake_init(decoded) == wire
        assert decoded.one_time_prekey_id == init.one_time_prekey_id

    def test_round_trip_without_one_time_prekey(self):
        _, _, init = make_init(False)
        decoded = decode_handshake_init(encode_handshake_init(init))
        assert decoded.one_time_prekey_id is None

    def test_decoded_init_verifies(self):
        _, _, init = make_init(True)
        decode_handshake_init(encode_handshake_init(init)).verify()

    def test_x25519_identity_is_derived_not_transmitted(self):
        """
        The X25519 form of the identity key is never on the wire.

        It is covered by the initiator's signature, and the receiver derives it.
        Transmitting it would only create a second copy that can disagree.
        """
        from src.crypto.key_management import identity_public_x25519

        _, _, init = make_init(True)
        wire = encode_handshake_init(init)
        derived = bytes(identity_public_x25519(init.identity_key))
        assert derived not in wire, 'x25519 identity key leaked onto the wire'

        decoded = decode_handshake_init(wire)
        assert bytes(decoded.identity_key_x25519) == derived

    def test_tampered_identity_key_breaks_verification(self):
        """
        A bit-flipped identity key may not even be a curve point.

        That must surface as a wire error, not as whatever exception libsodium
        happens to raise for a malformed point.
        """
        _, _, init = make_init(True)
        wire = bytearray(encode_handshake_init(init))
        wire[payload_offset(bytes(wire))] ^= 0xFF      # identity_key
        try:
            decoded = decode_handshake_init(bytes(wire))
        except WireError:
            return                                        # refused outright
        with pytest.raises(InvalidSignatureError):
            decoded.verify()

    def test_tampered_ephemeral_key_breaks_verification(self):
        _, _, init = make_init(True)
        wire = bytearray(encode_handshake_init(init))
        wire[payload_offset(bytes(wire)) + KEY_LEN] ^= 0xFF   # ephemeral_key
        decoded = decode_handshake_init(bytes(wire))
        with pytest.raises(InvalidSignatureError):
            decoded.verify()

    def test_tampered_signature_breaks_verification(self):
        _, _, init = make_init(True)
        wire = bytearray(encode_handshake_init(init))
        base = payload_offset(bytes(wire))
        wire[base + 2 * KEY_LEN + 63] ^= 0xFF          # signature
        decoded = decode_handshake_init(bytes(wire))
        with pytest.raises(InvalidSignatureError):
            decoded.verify()

    def test_truncation_at_every_offset_rejected(self):
        _, _, init = make_init(True)
        wire = encode_handshake_init(init)
        for cut in range(len(wire)):
            with pytest.raises(WireError):
                decode_handshake_init(wire[:cut])


class TestRatchetMessage:
    def _header(self):
        return {
            'dh': PrivateKey.generate().public_key.__bytes__().hex(),
            'pn': 0,
            'n': 7,
        }

    def test_round_trip(self):
        header = self._header()
        ciphertext = b'\x00' * 64
        wire = encode_ratchet_message(ciphertext, header)
        message = decode_ratchet_message(wire)
        assert message.dh.hex() == header['dh']
        assert message.pn == 0
        assert message.n == 7
        assert message.ciphertext == ciphertext
        assert message.to_header() == header

    def test_large_counters_round_trip(self):
        header = dict(self._header(), pn=2 ** 31, n=2 ** 32 - 1)
        message = decode_ratchet_message(encode_ratchet_message(b'\x00' * 40, header))
        assert message.pn == 2 ** 31
        assert message.n == 2 ** 32 - 1

    @pytest.mark.parametrize('field,value', [
        ('pn', -1), ('pn', 2 ** 32), ('n', -1), ('n', 2 ** 32),
    ])
    def test_out_of_range_counter_rejected(self, field, value):
        with pytest.raises(InvalidField):
            encode_ratchet_message(b'\x00' * 40, dict(self._header(), **{field: value}))

    def test_bad_dh_rejected(self):
        with pytest.raises(InvalidField):
            encode_ratchet_message(b'\x00' * 40, dict(self._header(), dh='abcd'))
        with pytest.raises(InvalidField):
            encode_ratchet_message(b'\x00' * 40, dict(self._header(), dh='zz' * 32))

    def test_short_ciphertext_rejected(self):
        wire = encode_ratchet_message(b'\x00' * 8, self._header())
        with pytest.raises(TruncatedFrame):
            decode_ratchet_message(wire)

    def test_truncation_at_every_offset_rejected(self):
        wire = encode_ratchet_message(b'\x00' * 64, self._header())
        for cut in range(len(wire)):
            with pytest.raises(WireError):
                decode_ratchet_message(wire[:cut])

    def test_counter_encoded_non_canonically_rejected(self):
        """
        A peer that pads its varints could otherwise produce a second encoding
        of the same header, and the header is authenticated as associated data.
        """
        payload = PrivateKey.generate().public_key.__bytes__() + b'\x80\x00' + b'\x00'
        wire = bytes((WIRE_VERSION, KIND_RATCHET_MESSAGE)) + encode_varint(len(payload)) + payload
        with pytest.raises(WireError):
            decode_ratchet_message(wire)


class TestUntrustedInput:
    """
    The decoders must be total. Anything other than WireError escaping is a bug
    in the boundary, whatever the input.
    """

    DECODERS = (decode_prekey_bundle, decode_handshake_init, decode_ratchet_message)

    def test_random_bytes(self):
        rng = random.Random(20260926)
        for _ in range(3000):
            blob = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 80)))
            for decoder in self.DECODERS:
                with pytest.raises(WireError):
                    decoder(blob)

    def test_bit_flips_in_valid_frames(self):
        _, responder = make_bundle(True)
        _, _, init = make_init(True)
        samples = [
            encode_prekey_bundle(responder.publish_bundle()),
            encode_handshake_init(init),
            encode_ratchet_message(
                b'\x00' * 64,
                {'dh': PrivateKey.generate().public_key.__bytes__().hex(), 'pn': 0, 'n': 1},
            ),
        ]
        rng = random.Random(7)
        for sample in samples:
            for _ in range(400):
                mutated = bytearray(sample)
                position = rng.randrange(len(mutated))
                mutated[position] ^= 1 << rng.randrange(8)
                blob = bytes(mutated)
                # Either it decodes, or it refuses; never anything else.
                for decoder in self.DECODERS:
                    try:
                        decoder(blob)
                    except WireError:
                        pass

    def test_valid_frames_are_accepted(self):
        """Control for the test above: the corpus really is decodable."""
        _, responder = make_bundle(True)
        _, _, init = make_init(True)
        assert decode_prekey_bundle(encode_prekey_bundle(responder.publish_bundle()))
        assert decode_handshake_init(encode_handshake_init(init))
        assert decode_ratchet_message(
            encode_ratchet_message(
                b'\x00' * 64,
                {'dh': PrivateKey.generate().public_key.__bytes__().hex(), 'pn': 0, 'n': 1},
            )
        )

    def test_oversized_frame_rejected_not_allocated(self):
        """
        A declared length beyond the limit must fail on the declaration itself.

        The point is that nothing is allocated for it, so this must not be
        written as a test that builds such a frame.
        """
        decoders = {
            KIND_PREKEY_BUNDLE: decode_prekey_bundle,
            KIND_HANDSHAKE_INIT: decode_handshake_init,
            KIND_RATCHET_MESSAGE: decode_ratchet_message,
        }
        for kind, decoder in decoders.items():
            header = bytes((WIRE_VERSION, kind)) + encode_varint(MAX_FRAME_BYTES + 1)
            with pytest.raises(FrameTooLarge):
                decoder(header)
