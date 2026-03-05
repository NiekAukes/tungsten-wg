density spagethi = add


add 1 2
add {left: 1, right: 2}
1 + 2
spagg<5>

// no dynamic arguments supported here
// const arguments are supported
density spagg<float abc> = add 1 2 # (100,33) // hashes are also comments, but specifically used for location

density function foo<float d>(float e1, float e3) {
    // GLSL-like code for the gpu
}

// @ symbols are used for metadata, attributes etc.
@preview
@gpu(bx = 8, by = 8, bz = 8)
density base3d = octave_noise {
    amplitudes: [0,1,1,1,2],
    first_octave: -7,
}







using minecraft;



add {
    left: mul {
        left: 4.0,
        right: quarter_negative {
            argument: mul {
                left: add {
                    left: overworld/depth, // single slashes can be used as path
                    right: mul {
                        left: overworld/jaggedness,
                        right: half_negative {
                            argument: noise {
                                noise: jagged,
                                xz_scale: 1500.0,
                                y_scale: 0.0
                            }
                        }
                    }
                },
                right: overworld/factor
            }
        }
    },
    right: overworld/base_3d_noise
}

density extra_jagged =
add
  mul
    4.0
    quarter_negative
      mul
       add
         minecraft:overworld/depth
         mul
           minecraft:overworld/jaggedness
           half_negative
             noise {
               noise: jagged,
               xz_scale: 1500.0,
               y_scale:0.0
             }
      overworld/factor
  overworld/base_3d_noise



density extra_jagged =
4.0 * quarter_negative
  mul
    add
      minecraft:overworld/depth
        minecraft:overworld/jaggedness * half_negative
          noise {
            noise: jagged,
            xz_scale: 1500.0,
            y_scale:0.0
          }
    overworld/factor
  + overworld/base_3d_noise

@identical(extra_jagged)
density function base_inline() {
    let depth = overworld/depth;
    let jaggedness = overworld/jaggedness;
    let factor = overworld/factor;

    let n = noise(jagged, 1500.0, 0.0);
    let jag = jaggedness * half_negative(n);

    let qn = quarter_negative(d + jag * factor);
    let left_side = 4.0 * qn;

    return left_side + overworld/base_3d_noise;
}
