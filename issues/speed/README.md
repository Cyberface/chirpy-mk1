# Speed Tests

Here we will assess the computational efficiency of the model.
Both compared to itself after various optimisations
and with respect to other models.

## To do

1. write a script to generate test data
   - `python gen-test-data.py` by default will create a
   high-mass and a low-mass in a new 'data' directory
   - `data/timings.txt` has the results for waveform
   generation time in seconds
2. write a script to generate new data
3. write a script to compare new data with test data


## tips and tricks

Currently chirpy-mk1 only knows about times.
It doesn't know about frequencies.
This means we cannot ask for the waveform from
a start frequency of say 30 Hz.

(To do this we need to invert the function f(t) to get t(f)...)

But without this we can simply generate an EOB waveform from
a given start frequency then we can take the time grid
that the EOB waveform is defined on and use that to
generate the chirpy-mk1 waveform.

This will actually give us an almost 1-to-1 speed comparison.
