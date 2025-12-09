## Notes on quantization

### How to represent numbers

each value in the number is a bit, each bit can be one of `sign`, `exponent`, `mantissa`.

1. float16(fp16) - also called half precision
(ref - https://en.wikipedia.org/wiki/Half-precision_floating-point_format)

for this, we have 
    - 1 bit for sign (S)
    - 5 bits for exponent ( E E E E E)
    - 10 bits for mantisssa (F F F F F F F F F F)

now, according to IEEE-754 convention, the exponents are stored in a biased form.

```
Estored = Eactual + bias

where bias = 2^(k − 1) − 1
k = exponent bits
```

for fp16, bias = 2 ^ 4 - 1 = 15

So for fp16 we shift them by +15 when storing, and then subtract 15 when reading.

Exponent range

1. minimum exponent bit -> 00000 (reserved for zeros)
2. maximum exponent bit -> 11111 (reserved for NaNs/infinity)

so the valid range is between 00001 and 11110 = 1 and 30

true exponent for 00001 = 1 - 15 = -14
true exponent for 11110 = 30 - 15 = 15

the final exponent range is `-14 < E < 15`

now, mantissa ~ 2, so the total range of values it can take on is 2 * 2^15 = 65504

Q. convert 13.5 to fp 16

-> 1101.1 in binary
-> normalize to the form 1.F x 2^E = 1.1011 x 2^3
    exponent = 3
    mantissa = 1.1011
    sign = 0 (since posiitve)

-> shift exponent by 15 = 15 + 3 = 18
-> exponent = 10010 in binary (18 needs 5 bits)
-> mantissa = 11011 00000

final representation
0 10010 1011000000

2. fp32(float32) - single/full precision

for this, we have 
    - 1 bit for sign (S)
    - 8 bits for exponent ( E E E E E E E E)
    - 24 bits for mantisssa (F F F F F F F F F F F F F F F F F F F F F F F F)

bias = 2^8-1 - 1 = 127

the range is `−126 < E <127`

3. bf16 (brain float)

for this, we have 
    - 1 bit for sign (S)
    - 8 bits for exponent ( E E E E E E E E)
    - 7 bits for mantisssa (F F F F F F F)

bias = 2^8-1 - 1 = 127

the range is `−126 < E <127`



random fact: how to calculate size of 70 billion params llm.

for fp32, we need 32/8 * 70 = 280 GB
for fp16 we need 16/8 * 70 = 140 GB