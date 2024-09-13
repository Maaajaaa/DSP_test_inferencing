# About

This is kind of a port of scottlawsonbc's (audio-reactive-led-strip)[https://github.com/scottlawsonbc/audio-reactive-led-strip]. However porting the Python rather meant extracting the principle of operation and getting similar algorithms to run directly on a low-power SoC, such as the ARM-M4-based NRF52480 as seeed studio offers it in a tiny form factor which included a pdm microphone as well a battery charge controller (see Examples/CollarExample).

# Limitations

For now, I've only implmented my favourite of the effects that  (audio-reactive-led-strip)[https://github.com/scottlawsonbc/audio-reactive-led-strip] offers, "Scroll". I've also only written one full project in  Examples/CollarExample, all the other examples still need to be adjusted and tested.

# Todo

- add more effects
- fix flickering when adjusting brightness/gain
- provide better documentation
- provide examples for further platforms and microphone types
- possibly push this into WLED, though the use of MFCC might be to compute-expensive for that use-case
- very far fetched, but possible: write addtional version, driven by a simply filtered FFT rather than the compute-expensive FFT+MFCC
