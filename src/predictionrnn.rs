use crate::structrnn::FunctionTrain;
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

pub fn predictiontrain_rnn(
    pathfile: &str,
    effect: &str,
    transcriptfile: &str,
    epochs: &str,
    learningrate: &str,
) -> Result<String> {
    let file = File::open(pathfile).expect("file not found");
    let fileread = BufReader::new(file);

    let transcript = File::open(transcriptfile).expect("file not found");
    let transcriptread = BufReader::new(transcript);

    let mut vecstore: Vec<FunctionTrain> = Vec::new();
    let mut fastastore_id: Vec<String> = Vec::new();
    let mut fastastore_seq: Vec<String> = Vec::new();

    for i in fileread.lines() {
        let line = i.expect("file not found");
        let linevec = line.split('\t').collect::<Vec<_>>();
        if linevec[5] == effect {
            vecstore.push(FunctionTrain {
                functionalt: linevec[4].to_string(),
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
