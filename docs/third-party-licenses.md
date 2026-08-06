# Third-party licenses

## purgedcv

`nakagai/stats.py` vendors four statistics (`probabilistic_sharpe_ratio`,
`deflated_sharpe_ratio`, `min_track_record_length`, `effective_n_trials`)
from [purged-cross-validation](https://github.com/eslazarev/purged-cross-validation)
(purgedcv), version 0.1.3. The maths is unchanged from the original; the
entry points were adapted to take a `PooledMoments` derived from pooled sums
instead of a raw returns array.

```
MIT License

Copyright (c) 2026 Evgenii Lazarev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
