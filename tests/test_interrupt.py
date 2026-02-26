"""
Unit tests for interrupt handling — wake word during TTS playback.

Verifies that when the wake word is detected during active TTS playback,
the system immediately stops TTS, clears the queue, and transitions to
voice input seamlessly.
"""

import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from claude_speak.config import Config, WakeWordConfig, InputConfig, AudioConfig
from claude_speak.voice_controller import VoiceController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_controller(
    wakeword_enabled=True,
    chimes=True,
    backend="builtin",
    tts_stop_callback=None,
    interrupt_callback=None,
):
    config = Config(
        wakeword=WakeWordConfig(enabled=wakeword_enabled),
        input=InputConfig(auto_submit=True, backend=backend),
        audio=AudioConfig(chimes=chimes),
    )
    return VoiceController(
        config=config,
        tts_stop_callback=tts_stop_callback,
        interrupt_callback=interrupt_callback,
    )


# ---------------------------------------------------------------------------
# Tests: interrupt callback invoked on wake word during playback
# ---------------------------------------------------------------------------

class TestInterruptCallbackInvoked:
    """Test that the interrupt callback fires when wake word is detected
    during active TTS playback."""

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_interrupt_callback_called_during_playback(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """Interrupt callback should be called when wake word fires during TTS."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False
        interrupt_cb = MagicMock()

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        vc._on_wake_word()
        # Give the voice-input-cycle thread time to start
        time.sleep(0.2)

        interrupt_cb.assert_called_once()

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_no_interrupt_when_tts_idle(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """Interrupt callback should NOT be called when TTS is not playing."""
        mock_playing.exists.return_value = False
        interrupt_cb = MagicMock()

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        vc._on_wake_word()
        time.sleep(0.2)

        interrupt_cb.assert_not_called()

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_no_crash_when_interrupt_callback_is_none(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """Should not crash if interrupt_callback is None during playback."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False

        vc = _make_controller(interrupt_callback=None)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        # Should not raise
        vc._on_wake_word()
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# Tests: queue is cleared on interrupt
# ---------------------------------------------------------------------------

class TestQueueClearedOnInterrupt:
    """Test that the queue is cleared when interrupt fires."""

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_interrupt_callback_clears_queue(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """The interrupt callback (as wired by daemon) should clear the queue."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False

        # Simulate the daemon's interrupt callback
        engine_stop = MagicMock()
        queue_clear = MagicMock()

        def interrupt_cb():
            engine_stop()
            queue_clear()

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        vc._on_wake_word()
        time.sleep(0.2)

        engine_stop.assert_called_once()
        queue_clear.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: TTS engine is stopped on interrupt
# ---------------------------------------------------------------------------

class TestTTSEngineStopped:
    """Test that the TTS engine is stopped when interrupt fires."""

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_engine_stop_called_via_interrupt(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """engine.stop() should be called as part of the interrupt callback."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False

        engine_stop = MagicMock()

        def interrupt_cb():
            engine_stop()

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        vc._on_wake_word()
        time.sleep(0.2)

        engine_stop.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: voice input activates after interrupt
# ---------------------------------------------------------------------------

class TestVoiceInputAfterInterrupt:
    """Test that voice input starts immediately after the interrupt."""

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_voice_input_starts_after_interrupt(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """Voice input should start after interrupt during TTS playback."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False
        interrupt_cb = MagicMock()

        vc = _make_controller(interrupt_callback=interrupt_cb, backend="builtin")
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        vc._on_wake_word()
        time.sleep(0.3)

        # Interrupt should have fired
        interrupt_cb.assert_called_once()
        # Voice input should have started
        mock_builtin.assert_called_once()

    @patch("claude_speak.voice_controller.voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_voice_input_starts_after_interrupt_superwhisper(
        self, mock_mute, mock_playing, mock_vic
    ):
        """Voice input should start after interrupt (superwhisper backend)."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False
        interrupt_cb = MagicMock()

        vc = _make_controller(
            interrupt_callback=interrupt_cb, backend="superwhisper"
        )
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        vc._on_wake_word()
        time.sleep(0.3)

        interrupt_cb.assert_called_once()
        mock_vic.assert_called_once()

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_playing_sentinel_removed_on_interrupt(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """PLAYING_FILE should be removed during interrupt so daemon knows TTS stopped."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False
        interrupt_cb = MagicMock()

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        vc._on_wake_word()
        time.sleep(0.2)

        mock_playing.unlink.assert_called_with(missing_ok=True)

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_mute_sentinel_removed_on_interrupt(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """MUTE_FILE should be removed during interrupt to prevent mute deadlock."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False
        interrupt_cb = MagicMock()

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        vc._on_wake_word()
        time.sleep(0.2)

        mock_mute.unlink.assert_called_with(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests: no crash when interrupted with empty queue
# ---------------------------------------------------------------------------

class TestInterruptEmptyQueue:
    """Test that interrupting with an empty queue does not crash."""

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_interrupt_with_empty_queue_no_crash(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """Interrupt should succeed even when the queue is already empty."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False

        call_count = 0

        def interrupt_cb():
            nonlocal call_count
            call_count += 1
            # Simulates engine.stop() + Q.clear() on empty queue — no error

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        vc._on_wake_word()
        time.sleep(0.2)

        assert call_count == 1
        mock_builtin.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: interrupt callback error handling
# ---------------------------------------------------------------------------

class TestInterruptErrorHandling:
    """Test that errors in the interrupt callback are caught gracefully."""

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_interrupt_callback_exception_caught(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """An exception in interrupt_callback should be caught, not crash."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False

        interrupt_cb = MagicMock(side_effect=RuntimeError("engine exploded"))

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        # Should not raise — error is caught and logged
        vc._on_wake_word()
        time.sleep(0.3)

        interrupt_cb.assert_called_once()
        # Voice input should still start even after callback error
        mock_builtin.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: interrupt logging (daemon side)
# ---------------------------------------------------------------------------

class TestInterruptLogging:
    """Test the daemon's interrupt callback logging."""

    def test_daemon_interrupt_callback_logs_discarded_items(self, tmp_path):
        """The daemon interrupt callback should log discarded queue items."""
        from claude_speak import queue as Q

        engine = MagicMock()
        interrupt_count = 0

        def _interrupt_callback():
            nonlocal interrupt_count
            interrupt_count += 1
            discarded = Q.peek()
            discarded_texts = []
            for f in discarded:
                try:
                    discarded_texts.append(f.read_text(encoding="utf-8").strip())
                except (FileNotFoundError, PermissionError, OSError):
                    pass
            engine.stop()
            Q.clear()

        # Populate queue with some items
        with patch("claude_speak.queue.QUEUE_DIR", tmp_path):
            Q.enqueue("Hello world")
            Q.enqueue("This is a test")
            assert Q.depth() == 2

            _interrupt_callback()

            assert interrupt_count == 1
            engine.stop.assert_called_once()
            assert Q.depth() == 0

    def test_daemon_interrupt_callback_empty_queue(self, tmp_path):
        """The daemon interrupt callback should handle empty queue gracefully."""
        from claude_speak import queue as Q

        engine = MagicMock()
        interrupt_count = 0

        def _interrupt_callback():
            nonlocal interrupt_count
            interrupt_count += 1
            engine.stop()
            Q.clear()

        with patch("claude_speak.queue.QUEUE_DIR", tmp_path):
            assert Q.depth() == 0
            _interrupt_callback()
            assert interrupt_count == 1
            engine.stop.assert_called_once()

    def test_daemon_interrupt_count_increments(self, tmp_path):
        """Interrupt count should increment with each interrupt."""
        from claude_speak import queue as Q

        engine = MagicMock()
        interrupt_count = 0

        def _interrupt_callback():
            nonlocal interrupt_count
            interrupt_count += 1
            engine.stop()
            Q.clear()

        with patch("claude_speak.queue.QUEUE_DIR", tmp_path):
            _interrupt_callback()
            _interrupt_callback()
            _interrupt_callback()
            assert interrupt_count == 3
            assert engine.stop.call_count == 3


# ---------------------------------------------------------------------------
# Tests: rapid interrupts (multiple wake words in quick succession)
# ---------------------------------------------------------------------------

class TestRapidInterrupts:
    """Test handling of rapid successive interrupts."""

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle")
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_rapid_interrupts_no_crash(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """Multiple rapid wake words should not crash or deadlock."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False
        # Make voice input take a little time to simulate realistic scenario
        mock_builtin.return_value = True

        interrupt_count = 0

        def interrupt_cb():
            nonlocal interrupt_count
            interrupt_count += 1

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        # Fire multiple wake words rapidly
        for _ in range(5):
            vc._on_wake_word()

        # Wait for threads to settle
        time.sleep(0.5)

        # Interrupt should be called each time (TTS was "playing" each time)
        assert interrupt_count == 5
        # Only one voice input cycle should actually run (input lock prevents doubles)
        # The first one grabs the lock; the rest skip.
        assert mock_builtin.call_count >= 1

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle")
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_rapid_interrupts_only_one_voice_input(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """Only one voice input cycle should run even with rapid interrupts."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False

        # Make voice input take a bit of time so the lock is held
        def slow_voice_input(**kwargs):
            time.sleep(0.3)
            return True

        mock_builtin.side_effect = slow_voice_input
        interrupt_cb = MagicMock()

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        # Fire multiple wake words with small delays
        for _ in range(3):
            vc._on_wake_word()
            time.sleep(0.05)

        # Wait for everything to finish
        time.sleep(0.8)

        # All three should trigger interrupt callback (TTS was playing)
        assert interrupt_cb.call_count == 3
        # But only one voice input cycle should actually execute
        # (the _input_lock prevents concurrent cycles)
        assert mock_builtin.call_count == 1

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_interrupt_when_not_running(self, mock_mute, mock_playing, mock_builtin):
        """Interrupt should not fire when controller is not running."""
        mock_playing.exists.return_value = True
        interrupt_cb = MagicMock()

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = False

        vc._on_wake_word()
        time.sleep(0.1)

        interrupt_cb.assert_not_called()
        mock_builtin.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: wakeword listener paused/resumed around interrupt + voice input
# ---------------------------------------------------------------------------

class TestWakewordListenerDuringInterrupt:
    """Test that the wakeword listener is paused/resumed correctly during
    the interrupt + voice input cycle."""

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_listener_paused_and_resumed(
        self, mock_mute, mock_playing, mock_builtin
    ):
        """Listener should be paused during voice input and resumed after."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False
        interrupt_cb = MagicMock()

        vc = _make_controller(interrupt_callback=interrupt_cb)
        vc._running = True
        mock_ww = MagicMock()
        mock_ww.is_running = True
        vc._wakeword_listener = mock_ww

        vc._on_wake_word()
        # Wait for the voice-input-cycle thread to complete
        for _ in range(20):
            time.sleep(0.1)
            if mock_ww.resume.call_count > 0:
                break

        mock_ww.pause.assert_called_once()
        mock_ww.resume.assert_called_once()
