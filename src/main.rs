mod args;
mod egraph;
mod egraphpop;
use crate::args::CommandParse;
use crate::args::Commands;
use crate::egraph::knnclassification;
use crate::egraph::logisticclassification;
use crate::egraph::randomforestclassification;
use crate::egraph::variantclasslabel_knn;
use crate::egraph::variantclasslabel_logistic;
use crate::egraph::variantclasslabel_random;
use crate::egraphpop::pop_knn;
use crate::egraphpop::pop_logistic;
use crate::egraphpop::pop_random;
use crate::egraphpop::popknnclassification;
use crate::egraphpop::poplogisticclassification;
use crate::egraphpop::poprandomforestclassification;
use clap::Parser;
use figlet_rs::FIGfont;
mod altrnn;
mod predictionrnn;
mod refrnn;
mod structrnn;
use crate::altrnn::alttrain_rnn;
use crate::predictionrnn::predictiontrain_rnn;
use crate::refrnn::reftrain_rnn;

/*
Gaurav Sablok
gsablok@proton.me
*/

fn main() {
    let fontgenerate = FIGfont::standard().unwrap();
    let repgenerate = fontgenerate.convert("eGRAPH");
    println!("{}", repgenerate.unwrap());
    let argparse = CommandParse::parse();
    match &argparse.command {
        Commands::Classifier {
            filepathinput,
            qualityinput,
            genotypeinput,
            thread,
        } => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .unwrap();
            pool.install(|| {
                let command_logistic =
                    logisticclassification(filepathinput, qualityinput, genotypeinput).unwrap();
                let command_knn =
                    knnclassification(filepathinput, qualityinput, genotypeinput).unwrap();
                let command_random =
                    randomforestclassification(filepathinput, qualityinput, genotypeinput).unwrap();
                println!(
                    "The commands have been finished:{}\n{}\n{}\n",
                    command_logistic, command_knn, command_random
                );
            });
        }
        Commands::VariantClassifier {
            variantinput,
            qualityvalueinput,
            fileinput,
            genotypepathinput,
            thread,
        } => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .unwrap();
            pool.install(|| {
                let command_knn = variantclasslabel_knn(
                    fileinput,
                    variantinput,
                    qualityvalueinput,
                    genotypepathinput,
                )
                .unwrap();
                let command_logistic = variantclasslabel_logistic(
                    fileinput,
                    variantinput,
                    qualityvalueinput,
                    genotypepathinput,
                )
                .unwrap();
                let command_random = variantclasslabel_random(
                    fileinput,
                    variantinput,
                    qualityvalueinput,
                    genotypepathinput,
                )
                .unwrap();
                println!(
                    "The commands have been finished:{}\n{}\n{}\n",
                    command_knn, command_logistic, command_random
                );
            });
        }
        Commands::PopulationVariant {
            directoryinput,
            variantinput,
            qualityvalue,
            genotypeprediction,
            thread,
        } => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .unwrap();
            pool.install(|| {
                let command_knn = pop_knn(
                    directoryinput,
                    variantinput,
                    qualityvalue,
                    genotypeprediction,
                )
                .unwrap();
                let command_log = pop_logistic(
                    directoryinput,
                    variantinput,
                    qualityvalue,
                    genotypeprediction,
                )
                .unwrap();
                let command_random = pop_random(
                    directoryinput,
                    variantinput,
                    qualityvalue,
                    genotypeprediction,
                )
                .unwrap();
                println!(
                    "The command for the all classifier have been finished:{}\t{}\t{}\n",
                    command_knn, command_log, command_random
                );
            });
        }
        Commands::Population {
            directoryinput,
            qualityinput,
            genotypeinput,
            thread,
        } => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread.parse::<usize>().unwrap())
                .build()
                .unwrap();
            pool.install(|| {
                let command_log =
                    poplogisticclassification(directoryinput, qualityinput, genotypeinput).unwrap();
                let command_knn =
                    popknnclassification(directoryinput, qualityinput, genotypeinput).unwrap();
                let command_random =
                    poprandomforestclassification(directoryinput, qualityinput, genotypeinput)
                        .unwrap();
                println!(
                    "The population variant classification has been trained:{}\t{}\t{}\n",
                    command_log, command_knn, command_random
                );
            });
        }
        Commands::RefAllele {
            evaifile,
            refallele,
            transcript,
            epoch,
            learningrate,
        } => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(10usize)
                .build()
                .expect("Threads failed");
            pool.install(|| {
                let rnntrain =
                    reftrain_rnn(evaifile, refallele, transcript, epoch, learningrate).unwrap();
                println!("The command has been finished:{}", rnntrain);
            });
        }
        Commands::AltAllele {
            evaifile,
            altallele,
            transcript,
            epoch,
            learningrate,
        } => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(10)
                .build()
                .expect("threads not found");
            pool.install(|| {
                let commands =
                    alttrain_rnn(evaifile, altallele, transcript, epoch, learningrate).unwrap();
                println!("The command has finished:{}", commands);
            })
        }
        Commands::GeneEffect {
            evaifile,
            effects,
            transcript,
            epoch,
            learningrate,
        } => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(10)
                .build()
                .expect("threads failed");
            pool.install(|| {
                let command =
                    predictiontrain_rnn(evaifile, effects, transcript, epoch, learningrate)
                        .unwrap();
                println!("The commands have failed:{}", command);
            });
        }
    }
}
