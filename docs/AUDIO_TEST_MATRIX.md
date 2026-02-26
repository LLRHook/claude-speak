# Claude-Speak Audio Test Matrix

**Document Purpose**: Manual QA checklist for verifying audio functionality across hardware configurations during release testing.

**Last Updated**: 2026-02-26
**Applicable to**: claude-speak v1.0+

---

## Quick Reference: Test Configuration Summary

| Configuration | Output | Input | Complexity | Priority |
|---|---|---|---|---|
| Built-in speakers + mic | Built-in speakers | Built-in mic | Low | Critical |
| Wired headphones (3.5mm) | 3.5mm jack | Built-in mic | Low | Critical |
| USB microphone | Built-in speakers | USB mic | Medium | High |
| AirPods (standard) | BT audio | Built-in mic | High | High |
| AirPods Pro | BT audio (ANC) | Built-in mic | High | High |
| AirPods Max | BT audio (spatial) | Built-in mic | High | Medium |
| Other BT headphones | BT audio | Built-in mic | High | High |
| External USB speakers | USB speakers | Built-in mic | Medium | Medium |
| HDMI audio | HDMI display | Built-in mic | Medium | Low |
| USB audio interface | USB interface | USB interface | High | Medium |
| Mixed: BT + built-in mic | BT headphones | Built-in mic | High | High |
| No audio devices | N/A | N/A | Low | Low |

---

## Configuration 1: Built-in Speakers + Built-in Microphone

**Platform**: MacBook (all models)
**Hardware**: MacBook internal speakers and integrated microphone

### Setup Steps

1. Ensure no external audio devices are connected
2. Verify System Preferences → Sound → Output shows "Internal Speakers"
3. Verify System Preferences → Sound → Input shows "Internal Microphone"
4. Launch claude-speak application
5. In app settings, confirm audio input and output are set to defaults (Internal Microphone/Internal Speakers)

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **TTS Output** | Trigger a voice response from claude-speak (e.g., wake word activation) | Audio plays through built-in speakers clearly | Sound is audible, no distortion, clear speech |
| **Wake Word Detection** | Say "Claude" or configured wake word near MacBook | Audio is captured, wake word is detected, LED/indicator responds | Detection succeeds 9/10 times, minimal false positives |
| **Voice Input** | After wake word, speak a command or question | Audio is captured through built-in mic, STT processes input | Speech is recognized, no audio clipping, latency < 2s |
| **Stop Command** | Say "stop" or configured stop command during playback | Audio playback stops immediately | Playback stops within 100ms |

### Expected Behavior

- TTS output should be balanced, not too loud or quiet
- Built-in mic captures speech without excessive background noise filtering
- Wake word detection is responsive (< 500ms response time)
- No audio feedback loops or echoing

### Known Issues

- Background noise (fans, keyboard) may affect wake word detection
- Volume level may be low on some MacBook models with older audio drivers
- Occasional audio dropout if system is under heavy load

### Pass/Fail Criteria

- [ ] All four test cases pass
- [ ] No crashes or exceptions during audio operations
- [ ] Response time for wake word detection < 1 second
- [ ] Audio quality is acceptable for demonstration purposes

---

## Configuration 2: Wired Headphones (3.5mm or USB-C Adapter)

**Hardware**: Standard wired headphones with 3.5mm jack or USB-C adapter

### Setup Steps

1. Connect headphones via 3.5mm jack (or USB-C adapter to USB-C port)
2. Verify System Preferences → Sound → Output shows headphones device
3. If using USB-C adapter, wait for device enumeration (2-3 seconds)
4. Launch claude-speak application
5. Verify app recognizes headphones as output device
6. For input, confirm built-in microphone is still selected (headphones typically don't have integrated mic)

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **TTS Output** | Trigger voice response | Audio plays through headphones clearly | Sound is clear in both ears, balanced volume, no crackling |
| **Wake Word Detection** | Say wake word near MacBook mic | Built-in mic captures audio, wake word triggers | Detection succeeds, normal latency |
| **Voice Input** | Speak command into built-in mic while wearing headphones | Audio from both output and input works simultaneously | Simultaneous I/O successful, no interference |
| **Stop Command** | Say stop command | Playback through headphones stops | Immediate stop, clean audio cutoff |

### Expected Behavior

- Audio is routed to headphones automatically upon connection
- Built-in mic continues to capture input for voice commands
- No audio latency or synchronization issues
- Headphone connection is stable during testing

### Known Issues

- Older USB-C adapters may have compatibility issues on newer MacBooks
- Some 3.5mm headphones may have impedance mismatch, causing low volume
- Connection may drop if adapter cable has poor contact

### Pass/Fail Criteria

- [ ] TTS audio is clear and appropriately leveled through headphones
- [ ] Built-in microphone input works correctly for voice detection
- [ ] No audio glitches or crackling during playback
- [ ] Device is recognized immediately upon connection

---

## Configuration 3: USB Microphone + Built-in Speakers

**Hardware**: External USB microphone (e.g., Blue Yeti, Audio-Technica AT2020, Rode NT-USB)

### Setup Steps

1. Connect USB microphone to available USB port
2. Wait for device to enumerate (3-5 seconds)
3. Open System Preferences → Sound
4. Verify Input device shows the USB microphone
5. Verify Output device shows "Internal Speakers"
6. Launch claude-speak application
7. In settings, select USB microphone as input device
8. Test microphone levels in System Preferences (should show green when speaking)

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **TTS Output** | Trigger voice response | Audio plays through built-in speakers | Clear output, appropriate volume level |
| **Wake Word Detection** | Speak wake word into USB mic | USB mic captures audio with good signal level | Detection succeeds with high confidence |
| **Voice Input (Quality)** | Speak naturally into USB mic | STT engine processes audio with high accuracy | Speech recognition is accurate, minimal errors |
| **Stop Command** | Say stop during playback | Playback stops, USB mic input continues | Clean stop, no audio artifacts |
| **Microphone Levels** | Monitor audio input levels during speech | Levels stay in green zone (not clipping in red) | No clipping, appropriate gain |

### Expected Behavior

- USB microphone provides higher quality audio than built-in mic
- Minimal background noise picked up due to better microphone design
- Wake word detection should be more reliable with USB mic
- Audio input levels are stable and consistent

### Known Issues

- Some USB microphones require drivers (check manufacturer's website)
- Microphone may have a "mute" button that disables input - verify it's not pressed
- USB power draw may be significant on some mics, causing voltage issues on underpowered hubs
- Background gain/noise reduction on some mics may be too aggressive

### Pass/Fail Criteria

- [ ] USB microphone is recognized by system and claude-speak
- [ ] Wake word detection is reliable (9+/10 success rate)
- [ ] Voice input has noticeably better quality than built-in mic
- [ ] No audio clipping or distortion during normal speaking volume
- [ ] Audio levels are stable throughout test session

---

## Configuration 4: AirPods (Standard)

**Hardware**: Apple AirPods (2nd generation or later)

### Setup Steps

1. Ensure AirPods are charged and in pairing mode
2. Open System Preferences → Bluetooth
3. Click "Connect" next to AirPods when they appear in the list
4. Wait for connection (5-10 seconds)
5. Verify System Preferences → Sound → Output shows "Your AirPods"
6. Verify System Preferences → Sound → Input shows "Internal Microphone" (AirPods input will be used for Siri, not voice commands in claude-speak)
7. Launch claude-speak application
8. Confirm output device is set to AirPods in app settings

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **TTS Output** | Trigger voice response | Audio plays through AirPods | Clear audio in both ears, balanced volume |
| **Wake Word Detection** | Say wake word near MacBook (built-in mic) | Built-in mic captures wake word | Detection succeeds, latency acceptable |
| **Voice Input** | Speak command into built-in mic while wearing AirPods | Input from built-in mic, output to AirPods | Simultaneous I/O works without interference |
| **Stop Command** | Say stop during playback | Playback stops | Immediate stop without lag |
| **Connection Stability** | Play audio for 60 seconds continuously | Connection remains stable | No dropouts, continuous playback |

### Expected Behavior

- Audio switches to AirPods when they connect
- Simultaneous audio input/output works without feedback
- Bluetooth connection is stable during use
- Audio quality is consistent throughout session
- Voice commands to built-in mic are captured without interference from AirPods audio

### Known Issues

- **Bluetooth Profile Switching**: macOS sometimes switches between A2DP (audio) and HFP (microphone) profiles. If HFP becomes active, the built-in mic may not be used
- **Microphone Input Conflict**: System may try to use AirPods mic instead of built-in mic; requires manual override in app settings
- **Connection Lag**: Initial connection may take 5-10 seconds, causing first audio playback to be delayed
- **Automatic Pause**: AirPods may pause playback when removed from ears (can be disabled in AirPods settings)

### Pass/Fail Criteria

- [ ] AirPods connect reliably via Bluetooth
- [ ] TTS audio plays clearly through AirPods
- [ ] Built-in microphone input for voice commands works while AirPods are playing audio
- [ ] No audio dropouts or connection interruptions during 60-second session
- [ ] Wake word detection works 8+/10 times with AirPods connected
- [ ] No system crashes or app hangs related to Bluetooth

---

## Configuration 5: AirPods Pro

**Hardware**: Apple AirPods Pro (1st or 2nd generation)

### Setup Steps

1. Ensure AirPods Pro are charged and in pairing mode
2. Open System Preferences → Bluetooth
3. Click "Connect" next to AirPods Pro when discovered
4. Wait for connection (5-10 seconds)
5. Verify System Preferences → Sound → Output shows "Your AirPods Pro"
6. Verify System Preferences → Sound → Input shows "Internal Microphone"
7. In System Preferences → Bluetooth → Options (AirPods Pro), configure:
   - Transparency mode: Leave as user preference (usually Off or Transparency)
   - Noise Control: Test both ANC enabled and disabled
8. Launch claude-speak application
9. In settings, confirm AirPods Pro as output device

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **TTS Output (ANC Off)** | Disable ANC, trigger voice response | Audio plays through AirPods Pro with clear highs/lows | Clear audio without over-processing |
| **TTS Output (ANC On)** | Enable ANC, trigger voice response | Audio plays with noise cancellation active | Audio quality unchanged, ANC doesn't interfere |
| **Wake Word Detection (ANC On)** | ANC enabled, say wake word near MacBook | Built-in mic captures audio, wake word detected | Detection works with ANC active |
| **Voice Input (Transparency On)** | Enable Transparency mode, speak into built-in mic | Ambient sound passes through, built-in mic captures | Voice input clear, environment audible |
| **Stop Command** | Say stop during playback with ANC on | Playback stops | Immediate stop |
| **Connection Stability** | Play audio for 120 seconds with ANC cycling | Connection stable across all modes | No dropouts, smooth transitions between ANC states |

### Expected Behavior

- AirPods Pro connect reliably with advanced features available
- ANC does not interfere with app functionality or built-in mic input
- Transparency mode allows environment audio while playing TTS
- Audio quality is exceptional across all features
- Built-in mic input is unaffected by ANC/Transparency settings

### Known Issues

- **ANC Impact on Latency**: Enabling ANC may add 50-100ms latency to audio processing
- **Transparency Mode Audio**: Transparency mode may pick up system audio and feed it back through built-in mic (rare)
- **Profile Switching**: Under heavy BT load, system may switch to HFP profile instead of A2DP
- **Battery Status**: Low battery on AirPods Pro may trigger unexpected disconnections

### Pass/Fail Criteria

- [ ] AirPods Pro connect and remain connected throughout session
- [ ] TTS audio is clear and high-quality with ANC both on and off
- [ ] Wake word detection and voice input work reliably with ANC enabled
- [ ] No audio dropouts during 120-second continuous playback
- [ ] Transparency mode does not interfere with built-in mic input
- [ ] Stop command is responsive regardless of ANC/Transparency state

---

## Configuration 6: AirPods Max

**Hardware**: Apple AirPods Max (spatial audio enabled)

### Setup Steps

1. Ensure AirPods Max are charged and in pairing mode
2. Open System Preferences → Bluetooth
3. Click "Connect" next to AirPods Max when discovered
4. Wait for connection (10-15 seconds, slightly longer than standard AirPods)
5. Verify System Preferences → Sound → Output shows "Your AirPods Max"
6. Verify System Preferences → Sound → Input shows "Internal Microphone"
7. In System Preferences → Bluetooth → Options (AirPods Max):
   - Spatial Audio: Enable for testing (if not already)
   - Transparency mode: Set preference for test environment
8. Launch claude-speak application
9. Confirm AirPods Max as output device

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **TTS Output (Spatial Disabled)** | Disable spatial audio, trigger voice response | Audio plays in standard stereo | Clear, balanced audio from both ears |
| **TTS Output (Spatial Enabled)** | Enable spatial audio, trigger voice response | Audio plays with spatial processing | Audio has spatial dimension, not disorienting |
| **Wake Word Detection** | Wearing AirPods Max, say wake word | Built-in mic captures audio through device | Detection succeeds 8+/10 times |
| **Voice Input (Spatial On)** | With spatial audio on, speak into built-in mic | Input captured, output processed spatially | Smooth simultaneous I/O |
| **Stop Command** | Say stop during spatial audio playback | Playback stops | Clean stop without spatial processing artifacts |
| **Passive Mode** | Test with AirPods Max in passive mode (ANC/Transparency off) | Basic audio output | No spatial effects, standard playback |

### Expected Behavior

- AirPods Max provide premium audio quality with extended frequency response
- Spatial audio processing does not interfere with voice functionality
- Connection is stable with advanced features (head tracking, spatial audio)
- Built-in mic continues to capture voice input reliably
- Audio latency is minimal despite spatial processing

### Known Issues

- **Spatial Audio Latency**: Spatial processing may add 30-50ms latency (not noticeable for TTS)
- **Heat During Extended Use**: AirPods Max can generate heat during long sessions; monitor for discomfort
- **Profile Switching**: Heavy Bluetooth load may force switch from A2DP to HFP, disabling spatial audio
- **Built-in Mic Interference**: Proximity sensors and spatial processing may occasionally interfere with built-in mic detection
- **Connection Complexity**: Multiple audio streams (spatial, head tracking) may cause slower pairing

### Pass/Fail Criteria

- [ ] AirPods Max connect and maintain stable connection for testing duration
- [ ] TTS audio is high-quality and clear with both spatial and non-spatial modes
- [ ] Wake word detection works reliably (8+/10) even with spatial audio enabled
- [ ] Voice input and audio output work simultaneously without interference
- [ ] Stop command is responsive in all modes
- [ ] No crashes or audio glitches during spatial audio playback

---

## Configuration 7: Other Bluetooth Headphones

**Hardware**: Sony WH-1000XM5, Bose QuietComfort Ultra, or similar BT headphones

### Setup Steps

1. Consult headphone manual for pairing mode (usually button press or combination)
2. Place headphones in pairing mode (LED indicates pairing mode)
3. Open System Preferences → Bluetooth
4. Click "Connect" when headphones appear in list
5. Complete pairing (usually requires confirmation on headphone device)
6. Verify System Preferences → Sound → Output shows headphone model
7. Verify System Preferences → Sound → Input shows "Internal Microphone"
8. Check headphone settings (via manufacturer app if available) for:
   - ANC/Noise Control: Test with enabled and disabled
   - Ambient/Transparency: Verify it doesn't interfere
   - EQ: Use neutral/default EQ for consistent testing
9. Launch claude-speak application
10. Confirm headphones selected as output device

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **TTS Output** | Trigger voice response | Audio plays through headphones | Clear, full-range audio |
| **Wake Word Detection** | Say wake word near MacBook | Built-in mic captures audio | Detection succeeds 7+/10 times |
| **Voice Input** | Speak command into built-in mic | Audio captured, processed normally | Good quality voice input |
| **ANC Impact** | Toggle ANC on/off, play audio | Connection stable in both modes | No dropouts or lag |
| **Stop Command** | Say stop during playback | Playback stops | Responsive stop |
| **Connection Stability** | Play audio continuously for 180 seconds | Connection remains stable | No disconnections or audio dropout |

### Expected Behavior

- Bluetooth connection is stable and responds to audio routing changes
- Audio quality meets expectations for headphone model
- Built-in mic input is not affected by headphone's BT connection
- No interference from headphone's ANC or other features
- Connection persists during full testing cycle

### Known Issues

- **Bluetooth Codec Variation**: Different headphones support different codecs (AAC, LDAC, aptX, SBC); audio quality varies
- **Driver Requirements**: Some BT headphones may require manufacturer-specific drivers or apps for full feature set
- **Multi-Device Switching**: Headphones may auto-switch to other paired devices (phone, tablet); disable in settings if possible
- **Microphone Input Switching**: System may attempt to use headphone's built-in mic instead of built-in mic; must override in app
- **Lag and Latency**: Cheaper BT headphones may have 100-200ms latency; premium models typically 30-50ms
- **Battery Sensitivity**: Low battery may trigger disconnections or quality degradation

### Pass/Fail Criteria

- [ ] Headphones pair and connect reliably via Bluetooth
- [ ] TTS audio is clear and appropriate quality for the headphone model
- [ ] Built-in microphone input works for voice commands (7+/10 success)
- [ ] No disconnections during 180-second continuous playback
- [ ] ANC or equivalent noise control does not interfere with app functionality
- [ ] Stop command is responsive
- [ ] Audio latency is acceptable (< 200ms for premium headphones)

---

## Configuration 8: External USB Speakers

**Hardware**: USB-powered external speakers (e.g., Bose SoundLink, UE Boom, or desktop USB speakers)

### Setup Steps

1. Connect USB speakers to available USB port
2. Power on speakers (if applicable)
3. Wait for device enumeration (3-5 seconds)
4. Open System Preferences → Sound → Output
5. Verify external USB speakers appear in device list
6. Select USB speakers as output device
7. Verify System Preferences → Sound → Input shows "Internal Microphone"
8. Launch claude-speak application
9. Confirm output device is set to USB speakers in app settings
10. Test volume control (both system and app volume should work)

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **TTS Output** | Trigger voice response | Audio plays through external speakers | Clear, full-range audio appropriate for room |
| **Volume Control** | Adjust volume via system and app controls | Volume changes smoothly, no clipping at high levels | Full range available, smooth control |
| **Wake Word Detection** | Say wake word near MacBook mic | Built-in mic captures audio | Detection succeeds 8+/10 times |
| **Voice Input** | Speak command into built-in mic | Audio captured while speakers play output | No feedback loop, input clean |
| **Stop Command** | Say stop during playback | Playback stops through speakers | Clean stop, no residual audio |
| **Extended Playback** | Play audio for 300+ seconds | Speakers remain on, connection stable | No thermal issues, consistent output |

### Expected Behavior

- Audio routing to USB speakers is automatic and immediate
- Speaker output is at appropriate volume for room-scale listening
- Built-in mic input is not affected by speaker output
- No audio feedback or interference
- Speaker connection is stable during extended use

### Known Issues

- **USB Power Draw**: Some USB speakers draw significant power, may require powered USB hub
- **Volume Muting**: Some USB speakers have hardware mute switch; ensure it's not activated
- **Mono vs Stereo**: Cheaper USB speakers may be mono or have poor stereo separation
- **Thermal Buildup**: Continuously powered USB speakers may heat up during extended testing
- **Driver Issues**: Some USB speaker models may not enumerate properly without specific drivers

### Pass/Fail Criteria

- [ ] USB speakers are recognized by system and app
- [ ] TTS audio is clear and at appropriate volume
- [ ] Wake word detection works 8+/10 times with speakers active
- [ ] Voice input (built-in mic) is clean with no feedback from speaker output
- [ ] Volume control is responsive and smooth
- [ ] No audio dropouts during extended playback (300+ seconds)

---

## Configuration 9: HDMI Audio (External Monitor)

**Hardware**: External display with HDMI audio support (e.g., 4K monitor with integrated speakers)

### Setup Steps

1. Connect external display via HDMI (or USB-C / Thunderbolt with video)
2. Wait for display enumeration and audio device recognition (5-10 seconds)
3. Open System Preferences → Sound → Output
4. Verify HDMI audio device appears (usually named "HDMI" or monitor model)
5. Select HDMI audio as output device
6. Verify System Preferences → Sound → Input shows "Internal Microphone"
7. Launch claude-speak application
8. Confirm output device is set to HDMI audio in app settings
9. Adjust display volume if it has physical controls

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **TTS Output** | Trigger voice response | Audio plays through monitor speakers | Audio audible, quality acceptable for small speakers |
| **Wake Word Detection** | Say wake word near MacBook mic | Built-in mic captures audio | Detection succeeds 8+/10 times |
| **Voice Input** | Speak into built-in mic while monitor plays audio | Input captured cleanly | No interference from monitor audio |
| **Stop Command** | Say stop during playback | Playback stops through monitor | Immediate stop |
| **Monitor Sleep Handling** | Put monitor to sleep during audio playback | Audio stops or reroutes appropriately | Graceful handling, no crashes |
| **HDMI Disconnection** | Unplug HDMI during app usage | System handles gracefully | Audio reroutes to default device or app handles error |

### Expected Behavior

- Audio device switches to HDMI immediately upon monitor connection
- Monitor speaker output is adequate for voice interaction (even if not high-quality)
- Built-in mic input functions normally with HDMI audio active
- Disconnecting monitor is handled gracefully without crashes
- Audio reroutes to internal speakers when HDMI is disconnected

### Known Issues

- **Monitor Speaker Quality**: Most external monitors have poor-quality built-in speakers; audio may sound tinny or thin
- **HDMI Handshake**: Some monitors may take 10-15 seconds to establish audio connection
- **Audio Device Switching**: System may not always switch audio back to internal speakers when HDMI is disconnected
- **Volume Control**: Monitor volume control may not be accessible via macOS system settings; physical buttons required
- **Wake on HDMI**: Some monitors disable audio when in standby; may need to wake monitor first

### Pass/Fail Criteria

- [ ] HDMI audio device is recognized and can be selected as output
- [ ] TTS audio plays through monitor speakers (even if poor quality)
- [ ] Wake word detection works 8+/10 times with HDMI audio active
- [ ] Voice input is clean and unaffected by monitor audio output
- [ ] Stop command is responsive
- [ ] HDMI disconnection is handled without crashes (audio reroutes or graceful error)

---

## Configuration 10: USB Audio Interface

**Hardware**: USB audio interface (e.g., Focusrite Scarlett, PreSonus Studio One, Behringer U-Phoria)

### Setup Steps

1. Connect USB audio interface via USB port
2. Power on interface (if applicable)
3. Wait for device enumeration (5-10 seconds)
4. Open System Preferences → Sound
5. Verify USB interface appears in both Output and Input device lists
6. Select USB interface as Output device
7. Select USB interface as Input device
8. Check audio interface settings:
   - Monitor level: Leave off to prevent feedback
   - Input gain: Set to appropriate level (should show in app or system)
   - Output level: Set to nominal level
9. Launch claude-speak application
10. Confirm Input and Output are set to USB interface in app settings
11. Run audio level test in system preferences

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **TTS Output** | Trigger voice response | Audio plays through interface outputs (speakers connected to interface) | Clear, high-quality audio |
| **Wake Word Detection (USB Mic)** | Speak wake word into USB interface mic input | USB interface captures audio, wake word detected | High-quality detection, minimal noise |
| **Voice Input (USB Mic)** | Speak command into USB interface mic input | Audio captured with excellent clarity | Clean signal, high SNR |
| **Input Level Monitoring** | Speak at normal volume into USB mic | Input levels stay in green, no clipping | No red clipping, appropriate gain |
| **Stop Command** | Say stop into USB mic | Playback stops | Responsive stop |
| **Full Audio Chain** | Complete interaction cycle through USB interface | Input and output work simultaneously | Smooth operation, no feedback |

### Expected Behavior

- USB audio interface provides professional-grade audio quality
- Input/output through interface is clean and noise-free
- Gain staging is appropriate for good signal-to-noise ratio
- No feedback loops or interference
- Interface enumeration is reliable

### Known Issues

- **Driver Requirements**: Most USB audio interfaces require manufacturer drivers; generic USB audio may have limited functionality
- **Monitor Level Feedback**: If monitor level is enabled on interface, feedback loop may occur
- **Input/Output Latency**: Audio interfaces may introduce small latency (20-50ms); not noticeable for voice interaction
- **Sample Rate Mismatches**: Interface may default to high sample rates (96kHz, 192kHz); app should resample
- **USB Bus Bandwidth**: High-channel-count interfaces may have compatibility issues with high-speed USB hubs
- **Cable Quality**: Poor USB cables can cause audio glitches; ensure quality cable

### Pass/Fail Criteria

- [ ] USB audio interface is recognized as both input and output device
- [ ] TTS audio through interface outputs is clear and high-quality
- [ ] Wake word detection using USB interface mic is reliable (9+/10)
- [ ] Voice input audio is clean with no clipping or noise
- [ ] Simultaneous input/output through interface works without feedback
- [ ] Stop command is responsive
- [ ] No audio glitches during extended use

---

## Configuration 11: Mixed Setup - Bluetooth Headphones Output + Built-in Microphone Input

**Hardware Combination**: Bluetooth headphones (any model) + MacBook built-in microphone

### Setup Steps

1. Pair and connect Bluetooth headphones (see Configuration 4-7 for specific models)
2. Open System Preferences → Sound
3. Set Output device to Bluetooth headphones
4. Set Input device to "Internal Microphone" (explicitly select to prevent system from switching to BT mic)
5. Verify Input shows "Internal Microphone" and is properly selected
6. Launch claude-speak application
7. In app settings:
   - Output device: Bluetooth headphones
   - Input device: Internal Microphone
8. Test both input and output before running full test cycle

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **TTS Output to BT Headphones** | Trigger voice response | Audio plays through Bluetooth headphones | Clear audio in headphones |
| **Wake Word Detection (Built-in Mic)** | Say wake word near MacBook | Built-in mic captures audio, wake word detected | Detection succeeds 8+/10 times |
| **Voice Input (Built-in Mic)** | Speak command into built-in mic (not headphone mic) | Audio captured from built-in mic, not headphones | Voice input is clean, from correct source |
| **No Feedback/Interference** | Speak into built-in mic while audio plays in headphones | No audio feedback or echo | Clean separation between input/output |
| **Stop Command** | Say stop into built-in mic | Playback stops in headphones | Immediate stop |
| **Simultaneous I/O Duration** | Run continuous I/O test for 300 seconds | Connection and input remain stable | No dropouts, consistent performance |
| **Input/Output Switching** | Switch between built-in speakers and BT headphones, then back | System correctly routes audio | No confusion about I/O devices |

### Expected Behavior

- Bluetooth headphones receive TTS output reliably
- Built-in microphone captures voice input independently
- No audio feedback or cross-contamination between input and output
- System correctly handles asymmetric I/O configuration
- Voice commands are processed accurately even with BT audio active

### Known Issues

- **Microphone Profile Conflict**: macOS may attempt to switch input to Bluetooth device mic when in call mode
- **Bluetooth Profile Switching**: System may switch from A2DP (audio) to HFP (handsfree) profile automatically
  - Mitigation: Disable "Phone Calls" or "Handsfree" mode in Bluetooth settings if possible
  - Check if "Show Bluetooth in menu bar" is enabled; use menu to manage connections
- **Input Source Detection**: Application must explicitly specify built-in mic to avoid auto-switching to BT mic
- **Audio Echo**: If system incorrectly routes input through BT mic, audio may be echoed back to headphones
- **Latency**: Input/output may have different latencies causing sync issues; typically not noticeable

### Pass/Fail Criteria

- [ ] Bluetooth headphones reliably receive TTS audio
- [ ] Built-in microphone input is used (not Bluetooth mic)
- [ ] Wake word detection succeeds 8+/10 times with mixed setup
- [ ] Voice input is clean and free from feedback
- [ ] Stop command is responsive
- [ ] No audio dropouts or Bluetooth disconnections during 300-second test
- [ ] No profile switching issues (stays in A2DP mode)
- [ ] Application correctly identifies I/O devices and doesn't attempt to switch inputs

---

## Configuration 12: No Audio Devices (Headless/Minimal)

**Hardware Configuration**: MacBook with no audio input or output devices available, OR audio disabled

### Setup Steps

1. **Option A - Fully Headless**:
   - Disconnect all audio devices (headphones, external speakers, USB audio)
   - Disable internal speakers in System Preferences → Sound → Output by selecting "No Output Device" if available
   - Disable internal microphone in System Preferences → Sound → Input
   - Note: May not be fully possible on all Macs; at minimum, disable audio device selection

2. **Option B - Audio Disabled (More Realistic)**:
   - Keep hardware connected but disable in System Preferences:
     - Sound → Output: Select a non-audio device or mute system output
     - Sound → Input: Set to device that doesn't exist or is disabled
   - Or use command line:
     ```bash
     # Disable built-in microphone input (requires admin)
     sudo osascript -e 'set volume input volume 0'
     ```

3. Launch claude-speak application

### Test Cases

| Test Case | Steps | Expected Behavior | Pass Criteria |
|---|---|---|---|
| **Graceful Degradation** | Attempt to trigger voice response | App detects no audio output, handles gracefully | No crash, informative error message |
| **Error Handling** | Try to record voice input | App detects no audio input, handles gracefully | No crash, error message to user |
| **UI Behavior** | Check UI state with no devices | UI may disable audio buttons or show warnings | Disabled controls or clear warnings |
| **App Recovery** | Reconnect audio device while app is running | App recognizes new device and resumes normally | Auto-detection and recovery |

### Expected Behavior

- Application does not crash when audio devices are unavailable
- User receives clear message about missing audio
- Application can continue running (if applicable) or gracefully shut down
- When audio device is reconnected, application resumes functionality
- No error messages in system logs related to audio initialization

### Known Issues

- **Graceful Fallback**: Not all apps handle missing audio gracefully; claude-speak should log warnings and disable audio features
- **Device Enumeration**: System may always report at least "No Device" option; true headless is rare
- **Error Messages**: Users may be confused if they don't understand why app isn't working; clear messaging is essential

### Pass/Fail Criteria

- [ ] Application does not crash when no audio devices are available
- [ ] User-facing error message clearly indicates missing audio
- [ ] No exceptions or unhandled errors in logs
- [ ] Application can recover if audio device is reconnected
- [ ] If app remains running, audio features are clearly disabled
- [ ] Graceful shutdown or fallback mode is implemented

---

## Test Execution Checklist

### Pre-Test Setup
- [ ] Confirm macOS version (supported versions: 11.0+)
- [ ] Ensure application is built and installed correctly
- [ ] Verify test environment is quiet (minimal background noise)
- [ ] Charge all wireless devices (AirPods, headphones, USB speakers)
- [ ] Close unnecessary applications to reduce system load
- [ ] Reset Bluetooth if any previous pairing issues occurred

### Per-Configuration Steps
- [ ] Read configuration setup instructions thoroughly
- [ ] Connect/configure hardware as specified
- [ ] Verify audio input/output in System Preferences
- [ ] Launch application and check audio device detection
- [ ] Run all test cases in sequence
- [ ] Document results in test matrix below
- [ ] Note any issues encountered
- [ ] Disconnect/disable hardware before moving to next configuration

### Post-Test Cleanup
- [ ] Disconnect all external audio devices
- [ ] Return audio settings to defaults
- [ ] Close application cleanly
- [ ] Reset Bluetooth if any devices remain connected
- [ ] Clear any test logs or temporary files

---

## Test Results Template

### Results Recording

Copy and complete this table for each test session:

```
Test Date: _______________
Tester Name: _______________
macOS Version: _______________
App Version: _______________

| Configuration | Pass/Fail | Notes | Issues |
|---|---|---|---|
| Built-in speakers + mic | [ ] P [ ] F | | |
| Wired headphones (3.5mm) | [ ] P [ ] F | | |
| USB microphone | [ ] P [ ] F | | |
| AirPods (standard) | [ ] P [ ] F | | |
| AirPods Pro | [ ] P [ ] F | | |
| AirPods Max | [ ] P [ ] F | | |
| Other BT headphones | [ ] P [ ] F | | |
| External USB speakers | [ ] P [ ] F | | |
| HDMI audio | [ ] P [ ] F | | |
| USB audio interface | [ ] P [ ] F | | |
| Mixed BT + built-in mic | [ ] P [ ] F | | |
| No audio devices | [ ] P [ ] F | | |
```

---

## Common Audio Issues and Troubleshooting

### Issue: Wake Word Not Detecting

**Symptoms**: Wake word detection fails or is inconsistent (< 5/10 success rate)

**Possible Causes**:
- Microphone is too far from speaker
- Background noise is too high
- Microphone gain is too low
- Microphone is muted or disabled

**Troubleshooting**:
1. Verify microphone is selected in System Preferences → Sound → Input
2. Speak louder and closer to microphone (12 inches / 30cm optimal distance)
3. Test microphone levels in System Preferences (green bar should show when speaking)
4. Check for mute buttons on microphone or headphones
5. Reduce background noise (fans, keyboard) during testing
6. Verify app can access microphone permissions (System Preferences → Security & Privacy → Microphone)
7. Restart application and try again

### Issue: Audio Output is Muted or Very Quiet

**Symptoms**: TTS output is inaudible or much quieter than expected

**Possible Causes**:
- System volume is muted or very low
- Application audio is muted
- Device-specific mute switch is enabled
- Output device is not selected correctly

**Troubleshooting**:
1. Check system volume in menu bar (should not be muted)
2. Verify output device is selected in System Preferences → Sound → Output
3. Check for physical mute buttons on headphones, speakers, or audio interface
4. Adjust application volume slider if available
5. Verify audio device is powered on (for USB speakers, external monitors, audio interfaces)
6. Try playing audio from another application (e.g., music player) to verify device functionality
7. Restart application and try again

### Issue: Bluetooth Headphones Disconnect or Drop Audio

**Symptoms**: Bluetooth connection drops frequently or audio dropouts occur

**Possible Causes**:
- Bluetooth interference from other devices
- Low battery on wireless device
- Bluetooth profile switching (A2DP to HFP)
- Distance or obstruction between Mac and headphones

**Troubleshooting**:
1. Verify battery level on Bluetooth device (charge if below 20%)
2. Move closer to Mac (within 30 feet / 10 meters)
3. Remove obstructions between Mac and Bluetooth device
4. Restart Bluetooth: System Preferences → Bluetooth → Turn Off and back On
5. "Forget" and re-pair the device: Bluetooth settings → Options → Forget
6. Disable other Bluetooth devices that might cause interference
7. Check if device is paired with multiple devices; disconnect from others
8. Enable Bluetooth Low Energy mode if available in device settings

### Issue: Microphone Input/Output Feedback Loop (Echo)

**Symptoms**: Hearing own voice echoed back through speakers or headphones

**Possible Causes**:
- Built-in mic is picking up speaker output
- Application is incorrectly routing input/output
- Audio interface monitor level is enabled
- Microphone and speaker are too close

**Troubleshooting**:
1. Increase distance between microphone and speaker
2. Ensure microphone and speaker are on opposite sides of room
3. In app settings, verify that input and output devices are correctly specified
4. For USB audio interfaces, disable "Monitor" or "Direct Monitoring" in interface software
5. Verify system is not routing input through the same device as output
6. Test with headphones instead of speakers to isolate issue
7. If using USB microphone, move it away from speakers and system

### Issue: USB Audio Device Not Recognized

**Symptoms**: USB audio device does not appear in System Preferences or in application

**Possible Causes**:
- Device is not powered on
- USB cable is not fully connected
- Device driver is not installed
- USB port is not functioning
- Device firmware is outdated

**Troubleshooting**:
1. Verify device is powered on (check LED, listen for power-up sound)
2. Fully disconnect and reconnect USB cable
3. Try different USB port on Mac (avoid USB hubs if possible; use direct connection)
4. Check manufacturer website for required drivers or firmware updates
5. Restart Mac to force re-enumeration of USB devices
6. Open System Preferences → Sound and wait 10 seconds for device to appear
7. In System Information → USB, verify device appears in bus
8. Try device on another Mac to verify it's not defective

### Issue: Audio Sync Issues Between Input and Output

**Symptoms**: Voice input and audio output seem out of sync, causing confusion during interaction

**Possible Causes**:
- High system latency
- Different sample rates for input/output
- Bluetooth latency (50-200ms on consumer devices)
- USB interface buffering

**Troubleshooting**:
1. Built-in audio typically has lowest latency; test with built-in speakers + mic first
2. Avoid mixing different audio interfaces (e.g., USB mic + Bluetooth speakers)
3. If using Bluetooth, use premium devices (lower latency)
4. For USB audio interfaces, check buffer size settings in interface software (smaller = lower latency)
5. Disable any background audio processing in System Preferences
6. Close other audio applications that might increase latency
7. Note: Some latency is normal and acceptable for voice interaction (user expects slight delay)

---

## Release Checklist

Before releasing a new version of claude-speak, complete this audio testing checklist:

- [ ] Configuration 1 (Built-in) - PASS
- [ ] Configuration 2 (Wired Headphones) - PASS
- [ ] Configuration 3 (USB Microphone) - PASS
- [ ] Configuration 4 (AirPods Standard) - PASS
- [ ] Configuration 5 (AirPods Pro) - PASS
- [ ] Configuration 6 (AirPods Max) - PASS
- [ ] Configuration 7 (Other BT Headphones) - PASS
- [ ] Configuration 8 (USB Speakers) - PASS
- [ ] Configuration 9 (HDMI Audio) - PASS
- [ ] Configuration 10 (USB Audio Interface) - PASS
- [ ] Configuration 11 (Mixed BT + Built-in Mic) - PASS
- [ ] Configuration 12 (No Audio Devices) - PASS (graceful degradation)

**Overall Result**: [ ] Ready for Release [ ] Additional Testing Required [ ] Blocker Issues Found

**Issues Found**:
- [ ] Critical (blocks release)
- [ ] Major (should fix before release)
- [ ] Minor (can be deferred)

**Sign-off**:
- Tested by: _______________
- Date: _______________
- Version tested: _______________

---

## Notes for Future Enhancement

This document covers manual testing. Future improvements might include:
- Automated integration tests with mock audio devices
- Platform-specific testing for Windows and Linux
- Audio quality metrics (SNR, latency measurement)
- Stress testing with rapid device switching
- Performance profiling under high system load
- Internationalization testing with different languages/accents for wake word

---

**Document Version**: 1.0
**Last Updated**: 2026-02-26
**Maintained by**: QA Team
