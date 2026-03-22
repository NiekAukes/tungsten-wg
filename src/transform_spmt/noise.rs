use crate::{
    parse::model::NormalNoise,
    spmt::model::{
        BinaryOperator, DensityFunction, Expression, Function, PermutationTableInput, Statement,
        Var, Variable, VariableType,
    },
    transform_spmt::{density::make_rpos3, newvar},
};

const DOUBLE_PERLIN_NOISE_AMPLITUDE: f64 = 0.16666666666666666;
const DOUBLE_PERLIN_OFFSET: f64 = 1.0181268882175227;

pub fn lower_normal_noise<'m>(
    arena: &'m bumpalo::Bump,
    noise: NormalNoise,
    permutation_name: &str,
    cname: String,
    scale: (f32, f32, f32),
    as_density: bool,
) -> (Function<'m>, Vec<PermutationTableInput>) {
    let mut variables = Vec::new();
    let mut body = Vec::new();

    // create a random number seed generator

    /* -----------------------------
    Input position
    ----------------------------- */
    let rpos3 = newvar(arena, "rpos3", VariableType::Vec3);
    if as_density {
        let origin = newvar(arena, "origin", VariableType::Vec3);
        let pos3 = newvar(arena, "pos3", VariableType::Pos3);
        variables.push(rpos3.clone());
        body.push(Statement::Assign {
            target: rpos3.clone(),
            value: make_rpos3(pos3.clone(), origin.clone(), scale),
        });
    }

    // create perlin rpos3fxs for each octave
    // let rpos3f0 = rpos3 * freq0
    let frequencies = filtered_frequency_amplitude_list(noise);
    let mut rpos3fxs = Vec::new();
    for (i, (freq, _, _)) in frequencies.iter().enumerate() {
        let rpos3f = newvar(arena, &format!("rpos3f{}", i), VariableType::Vec3);
        body.push(Statement::Assign {
            target: rpos3f.clone(),
            value: Expression::BinaryOp {
                op: BinaryOperator::Multiply,
                left: Box::new(Expression::Variable(rpos3.clone())),
                right: Box::new(Expression::Float(*freq)),
            },
        });
        rpos3fxs.push(rpos3f.clone());
        variables.push(rpos3f);
    }

    // call perlin for each octave, and additionally add another perlin call with the same frequency but offset position
    // n[i] = (perlin(rpos3fi) + perlin(rpos3fi * scaling)) * amp[i]
    let normal_amplitude = double_perlin_amplitude(noise);
    let mut noise_terms = Vec::new();
    let mut permutation_table_inputs = Vec::new();
    for (i, (_, amp, octave_id)) in frequencies.iter().enumerate() {
        let n = newvar(arena, &format!("n{}", octave_id), VariableType::F32);

        // let noise_ident = random::xoroshiro_seed(&cname);
        // let perlin_ident = random::xoroshiro_seed(&format!("octave_{}", octave_id));
        let perm1 = PermutationTableInput {
            ident: permutation_name.to_string(),
            subident: Some(format!("octave_{}", octave_id)),
            subident_index: 0,
        };
        let perm2 = PermutationTableInput {
            ident: permutation_name.to_string(),
            subident: Some(format!("octave_{}", octave_id)),
            subident_index: 1,
        };
        let perm1_var = Expression::PermutationTable(perm1.clone());
        let perm2_var = Expression::PermutationTable(perm2.clone());
        permutation_table_inputs.push(perm1);
        permutation_table_inputs.push(perm2);
        let perlin1 = Expression::ExternCall {
            function_name: "perlin".into(),
            parameters: vec![Expression::Variable(rpos3fxs[i].clone()), perm1_var],
            parameter_types: vec![VariableType::Vec3, VariableType::PermutationTable],
        };
        let scaled_rpos3f = Expression::BinaryOp {
            op: BinaryOperator::Multiply,
            left: Box::new(Expression::Variable(rpos3fxs[i].clone())),
            right: Box::new(Expression::Float(DOUBLE_PERLIN_OFFSET)),
        };
        let perlin2 = Expression::ExternCall {
            function_name: "perlin".into(),
            parameters: vec![scaled_rpos3f, perm2_var],
            parameter_types: vec![VariableType::Vec3, VariableType::PermutationTable],
        };
        let noise_sum = Expression::BinaryOp {
            op: BinaryOperator::Add,
            left: Box::new(perlin1),
            right: Box::new(perlin2),
        };
        let scaled_noise = Expression::BinaryOp {
            op: BinaryOperator::Multiply,
            left: Box::new(noise_sum),
            right: Box::new(Expression::Float(*amp)),
        };
        body.push(Statement::Assign {
            target: n.clone(),
            value: scaled_noise,
        });
        noise_terms.push(Expression::Variable(n.clone()));
        variables.push(n);
    }

    // sum all noise terms
    let sum = noise_terms
        .into_iter()
        .reduce(|a, b| Expression::BinaryOp {
            op: BinaryOperator::Add,
            left: Box::new(a),
            right: Box::new(b),
        })
        .unwrap();

    let final_sum = Expression::BinaryOp {
        op: BinaryOperator::Multiply,
        left: Box::new(sum),
        right: Box::new(Expression::Float(normal_amplitude)),
    };
    body.push(Statement::Return(final_sum));

    (
        Function {
            canonical_name: Some(cname),
            //density_inputs: vec![],
            //helper_functions: vec![],
            parameters: vec![rpos3],
            body,
            variables,
        },
        permutation_table_inputs,
    )
}

fn filtered_frequency_amplitude_list(noise: NormalNoise) -> Vec<(f64, f64, i32)> {
    // get the base frequency from the first octave
    let mut freqs = Vec::new();
    let mut freq = 2.0f64.powi(noise.first_octave);
    let mut persistence = calc_persistence(noise.amplitudes.len() as i32);
    for i in 0..noise.amplitudes.len() {
        // if the amplitude for this octave is 0, skip it
        if noise.amplitudes[i] != 0.0 {
            let l = noise.first_octave + i as i32;
            freqs.push((freq, noise.amplitudes[i] * persistence, l));
        }
        freq *= 2.0;
        persistence /= 2.0;
    }
    freqs
}

/*
private DoublePerlinNoiseSampler(Random random, DoublePerlinNoiseSampler.NoiseParameters parameters, boolean modern) {
        int i = parameters.firstOctave;
        DoubleList doubleList = parameters.amplitudes;
        this.parameters = parameters;
        if (modern) {
            this.firstSampler = OctavePerlinNoiseSampler.create(random, i, doubleList);
            this.secondSampler = OctavePerlinNoiseSampler.create(random, i, doubleList);
        } else {
            this.firstSampler = OctavePerlinNoiseSampler.createLegacy(random, i, doubleList);
            this.secondSampler = OctavePerlinNoiseSampler.createLegacy(random, i, doubleList);
        }

        int j = Integer.MAX_VALUE;
        int k = Integer.MIN_VALUE;
        DoubleListIterator doubleListIterator = doubleList.iterator();

        while (doubleListIterator.hasNext()) {
            int l = doubleListIterator.nextIndex();
            double d = doubleListIterator.nextDouble();
            if (d != 0.0) {
                j = Math.min(j, l);
                k = Math.max(k, l);
            }
        }

        this.amplitude = 0.16666666666666666 / createAmplitude(k - j);
        this.maxValue = (this.firstSampler.getMaxValue() + this.secondSampler.getMaxValue()) * this.amplitude;
    }
*/

fn double_perlin_amplitude(noise: NormalNoise) -> f64 {
    let (min_octave, max_octave) = calc_octave_range(noise);
    DOUBLE_PERLIN_NOISE_AMPLITUDE / create_amplitude(max_octave - min_octave)
}

/*
private static double createAmplitude(int octaves) {
        return 0.1 * (1.0 + 1.0 / (octaves + 1));
    }
     */

fn calc_octave_range(noise: NormalNoise) -> (i32, i32) {
    let mut min_octave = i32::MAX;
    let mut max_octave = i32::MIN;

    for (i, amp) in noise.amplitudes.iter().enumerate() {
        if *amp != 0.0 {
            min_octave = min_octave.min(i as i32);
            max_octave = max_octave.max(i as i32);
        }
    }

    (min_octave, max_octave)
}

fn create_amplitude(octaves: i32) -> f64 {
    0.1 * (1.0 + 1.0 / ((octaves as f64) + 1.0))
}

// Math.pow(2.0, i - 1) / (Math.pow(2.0, i) - 1.0)

fn calc_persistence(amplitudes_count: i32) -> f64 {
    (2.0f64.powi(amplitudes_count - 1)) / (2.0f64.powi(amplitudes_count) - 1.0)
}
