use crate::structrnn::AltTrain;
use axonml::nn::functional::cross_entropy;
use axonml::nn::{self, Conv1d, Module, Sequential};
use axonml::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
*/

pub struct DNAAutoEncoder {
    encoder: Sequential,
    decoder: Sequential,
}

impl DNAAutoEncoder {
    pub fn new() -> Self {
        Self {
            encoder: nn::Sequential::new()
                .add(Conv1d::new(4, 16, 4))
                .add(nn::ReLU)
                .add(Conv1d::new(16, 32, 4))
                .add(nn::ReLU)
                .add(Conv1d::new(32, 64, 4)),

            decoder: nn::Sequential::new()
                .add(Linear::new(64, 32))
                .add(nn::ReLU)
                .add(Conv1d::new(32, 16, 4))
                .add(nn::ReLU)
                .add(Conv1d::new(16, 4, 4)),
        }
    }
}

impl Module for DNAAutoEncoder {
    fn forward(&self, input: &Variable) -> Variable {
        let variablespace = self.encoder.forward(input);
        self.decoder.forward(&variablespace)
    }
}

pub fn alttrain_rnn(
    pathfile: &str,
    altallele: &str,
    transcriptfile: &str,
    epochs: &str,
    learningrate: &str,
) -> Result<String> {
    let file = File::open(pathfile).expect("file not found");
    let fileread = BufReader::new(file);

    let transcript = File::open(transcriptfile).expect("file not found");
    let transcriptread = BufReader::new(transcript);

    let mut vecstore: Vec<AltTrain> = Vec::new();
    let mut fastastore_id: Vec<String> = Vec::new();
    let mut fastastore_seq: Vec<String> = Vec::new();

    for i in fileread.lines() {
        let line = i.expect("file not found");
        let linevec = line.split('\t').collect::<Vec<_>>();
        if linevec[4] == altallele {
            vecstore.push(AltTrain {
                altallele: linevec[4].to_string(),
                id: linevec[6].to_string(),
            });
        }
    }

    for i in transcriptread.lines() {
        let line = i.expect("line not found");
        if line.starts_with('>') {
            let linestring = line.split('>').collect::<Vec<_>>()[0].to_string();
            fastastore_id.push(linestring);
        } else {
            fastastore_seq.push(line);
        }
    }

    /*
    if the sequence contains the variant alt allele then label as 1
    if it doesnt contains the variant then label all of them as 0
    */

    let mut altrnn: Vec<String> = Vec::new();
    let mut labelaltrnn: Vec<f32> = Vec::new();

    for i in vecstore.iter() {
        for val in 0..fastastore_id.len() {
            if i.id == fastastore_id[val] {
                altrnn.push(fastastore_seq[val].clone());
                labelaltrnn.push(1f32);
            } else {
                altrnn.push(fastastore_seq[val].clone());
                labelaltrnn.push(0f32);
            }
        }
    }

    let splicestring = samelength(altrnn)?;

    let mut inputsvec = Vec::new();
    for i in splicestring.iter() {
        inputsvec.push(onehot_encode(i));
    }

    let model = DNAAutoEncoder::new();
    let batch_size = inputsvec.len() as usize;
    let seqlen = inputsvec[0].len() as usize;

    let flat_inputs: Vec<f32> = inputsvec.into_iter().flatten().collect();
    let inputs = Tensor::from_vec(flat_inputs, &[batch_size, 4, seqlen]).unwrap();
    let inputvar = Variable::new(inputs, false);

    let targets = Tensor::from_vec(labelaltrnn, &[batch_size, 1]).unwrap();
    let targetsvar = Variable::new(targets, false);

    let lr = learningrate.parse::<f32>().unwrap();
    let mut optimizer = Adam::new(model.parameters(), lr);

    let epochs_n = epochs.parse::<usize>().unwrap();
    for epoch in 1..=epochs_n {
        optimizer.zero_grad();
        let prediction = model.forward(&inputvar);
        let loss = cross_entropy(&prediction, &targetsvar);
        loss.backward();
        optimizer.step();
        println!("Epoch [{}/{}],", epoch, epochs_n);
    }

    Ok("The model has finished learning".to_string())
}

pub fn onehot_encode(seq: &str) -> Vec<f32> {
    let mut encoded = vec![0.0; 4 * seq.len()];
    for (i, c) in seq.chars().enumerate() {
        let row = match c {
            'A' | 'a' => 0,
            'C' | 'c' => 1,
            'G' | 'g' => 2,
            'T' | 't' => 3,
            _ => continue,
        };
        encoded[row * seq.len() + i] = 1.0;
    }
    encoded
}

pub fn samelength(strvec: Vec<String>) -> Result<Vec<String>> {
    let veclength = strvec.iter().map(|x| x.len()).collect::<Vec<_>>();
    let minvalue = veclength.iter().min().unwrap();
    let mut revisedstring: Vec<String> = Vec::new();

    for i in strvec.iter() {
        let splicedstring = &i[0..*minvalue];
        revisedstring.push(splicedstring.to_string());
    }
    Ok(revisedstring)
}

/*
 * First I wrote the image based and then used that as a skeleton to put the DNAseq based.
 *
 *
 *
//! Complete autoencoder example built on AxonML (https://github.com/AutomataNexus/AxonML),
//! a PyTorch-equivalent deep learning framework written in pure Rust.
//!
//! Architecture:
//!   Input (784) -> Encoder -> Latent (32) -> Decoder -> Reconstruction (784)
//!
//! Encoder: 784 -> 128 -> 32           (Linear + ReLU, bottleneck)
//! Decoder: 32  -> 128 -> 784 -> Sigmoid  (reconstructs values in [0, 1])
//!
//! Trained with MSE reconstruction loss and Adam.
//!
//! NOTE: AxonML is a fast-moving, community-maintained crate. The exact method
//! signatures below follow the patterns documented in its README (Sequential,
//! Module, Adam, DataLoader). If a signature has drifted since this was written,
//! check https://docs.rs/axonml for the current API and adjust field/method names
//! accordingly — the overall structure (encoder/decoder split, training loop,
//! reconstruction loss) will still apply.

use axonml::prelude::*;
use axonml_nn::{Linear, Module, ReLU, Sigmoid, Sequential, MSELoss};
use axonml_optim::{Adam, Optimizer};

/// An autoencoder composed of two Sequential sub-networks so the latent
/// (bottleneck) representation stays reachable via `encode`.
struct AutoEncoder {
    encoder: Sequential,
    decoder: Sequential,
}

impl AutoEncoder {
    fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        let encoder = Sequential::new()
            .add(Linear::new(input_dim, hidden_dim))
            .add(ReLU)
            .add(Linear::new(hidden_dim, latent_dim));

        let decoder = Sequential::new()
            .add(Linear::new(latent_dim, hidden_dim))
            .add(ReLU)
            .add(Linear::new(hidden_dim, input_dim))
            .add(Sigmoid);

        Self { encoder, decoder }
    }

    /// Encode input into the latent bottleneck.
    fn encode(&self, x: &Tensor<f32>) -> Tensor<f32> {
        self.encoder.forward(x)
    }

    /// Decode a latent vector back into input space.
    fn decode(&self, z: &Tensor<f32>) -> Tensor<f32> {
        self.decoder.forward(z)
    }

    /// Full forward pass: encode then decode.
    fn forward(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let z = self.encode(x);
        self.decode(&z)
    }

    /// All trainable parameters, for handing to the optimizer.
    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }

    fn train(&mut self) {
        self.encoder.train();
        self.decoder.train();
    }

    fn eval(&mut self) {
        self.encoder.eval();
        self.decoder.eval();
    }
}

fn main() {
    // --- Config ---
    let input_dim = 784; // e.g. flattened 28x28 images
    let hidden_dim = 128;
    let latent_dim = 32;
    let batch_size = 64;
    let num_batches = 50; // batches per epoch (synthetic data stand-in for a real dataset)
    let epochs = 20;
    let learning_rate = 1e-3;

    // --- Model & optimizer ---
    let mut model = AutoEncoder::new(input_dim, hidden_dim, latent_dim);
    let mut optimizer = Adam::new(model.parameters(), learning_rate);
    let loss_fn = MSELoss::new();

    model.train();

    // --- Training loop ---
    // Swap this synthetic data generator for a real Dataset/DataLoader
    // (e.g. axonml_vision::datasets::Mnist) when you have real data to reconstruct.
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for _ in 0..num_batches {
            // Synthetic batch: replace with real samples flattened to [batch, input_dim]
            let batch = rand::<f32>(&[batch_size, input_dim]);

            let reconstruction = model.forward(&batch);
            let loss = loss_fn.compute(&reconstruction, &batch);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item();
        }

        let avg_loss = epoch_loss / num_batches as f32;
        println!("epoch {:>3} | avg reconstruction MSE: {:.6}", epoch + 1, avg_loss);
    }

    // --- Inference / inspection ---
    model.eval();

    let sample = rand::<f32>(&[1, input_dim]);
    let latent = model.encode(&sample);
    let reconstruction = model.decode(&latent);

    println!("\nlatent shape:         {:?}", latent.shape());
    println!("reconstruction shape: {:?}", reconstruction.shape());

    // Save the trained weights for later inference.
    // save(&model.parameters(), "autoencoder.axonml").unwrap();
}

*/
