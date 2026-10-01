use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "egraph",
    version = "1.0",
    about = "classification for variant annotation.
       ************************************************
       Author Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Classify a single population
    Classifier {
        /// inputfile
        filepathinput: String,
        /// quality
        qualityinput: String,
        /// genotype prediction
        genotypeinput: String,
        /// thread for the analysis
        thread: String,
    },
    /// filter and classify the variants on the specific variant types for a single population
    VariantClassifier {
        /// file to be used
        fileinput: String,
        /// variant to be filtered
        variantinput: String,
        /// quality value
        qualityvalueinput: String,
        /// genotype prediction
        genotypepathinput: String,
        /// thread for the analysis
        thread: String,
    },
    /// Classify on an entire population
    Population {
        /// input directory
        directoryinput: String,
        /// qualityinput
        qualityinput: String,
        /// genotype prediction
        genotypeinput: String,
        /// thread for the analysis
        thread: String,
    },
    /// Classify an entire population over a specific variant
    PopulationVariant {
        /// input directory
        directoryinput: String,
        /// variant to be filtered
        variantinput: String,
        /// quality value input
        qualityvalue: String,
        /// genotype prediction
        genotypeprediction: String,
        /// thread for the analysis
        thread: String,
    },
    /// Ref Alt Neural Network
    RefAllele {
        /// path to the eVai file
        evaifile: String,
        /// name of the ref allele
        refallele: String,
        /// path to the transcript file
        transcript: String,
        /// number of epochs
        epoch: String,
        /// learning rate
        learningrate: String,
    },
    /// Alt Alt Neural Network
    AltAllele {
        /// path to the eVai file
        evaifile: String,
        /// name of the alt allele
        altallele: String,
        /// path to the transcript file
        transcript: String,
        /// number of epochs
        epoch: String,
        /// learning rate
        learningrate: String,
    },
    /// ref Alt Neural Network
    GeneEffect {
        /// path to the eVai file
        evaifile: String,
        /// gene effect such as intron_variant and others
        effects: String,
        /// path to the transcript file
        transcript: String,
        /// number of epochs
        epoch: String,
        /// learning rate
        learningrate: String,
    },
}
