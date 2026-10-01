use rand::SeedableRng;
use rand::rngs::StdRng;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::accuracy;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::path::Path;

/*
Gaurav Sablok
gsablok@proton.me
*/

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct AutoencoderData {
    pub nameallele: String,
    pub gentoype_54: f64,
    pub quality_55: f64,
    pub ro_number: f64,
}

type EGRAPHTYPE = Vec<AutoencoderData>;

type DIRETURN = Vec<(String, String, f64, f64)>;

/*
Reading an entire population and making a tensor type data for the same.
*/

pub fn readdata(path: &str) -> Result<DIRETURN, Box<dyn std::error::Error>> {
    let pathdir = Path::new(path);
    let mut returnvector: Vec<(String, String, f64, f64)> = Vec::new();
    for i in fs::read_dir(pathdir)? {
        let pathdir = i?.path();
        let filename = pathdir.to_str().unwrap();
        let fileopen = File::open(filename).expect("file not present");
        let fileread = BufReader::new(fileopen);
        for i in fileread.lines() {
            let line = i.expect("line not present");
            let linevec = line.split("\t").collect::<Vec<_>>();
            let mutabletuple: (String, String, f64, f64) = (
                linevec[4].to_string(),
                linevec[54].to_string(),
                linevec[55].parse::<f64>().unwrap(),
                linevec[58].parse::<f64>().unwrap(),
            );
            returnvector.push(mutabletuple);
        }
    }
    Ok(returnvector)
}

/*
  Entire collated data is converted into a single tensor.
*/

pub fn readdir_tensor(pathfile: &str) -> Result<EGRAPHTYPE, Box<dyn Error>> {
    let mutablevariable = readdata(pathfile).unwrap();
    let mut tensorvec: Vec<AutoencoderData> = Vec::new();
    for i in mutablevariable.iter() {
        if i.1 == "pass" {
            tensorvec.push(AutoencoderData {
                nameallele: i.0.to_string().clone(),
                gentoype_54: i.2,
                quality_55: i.3,
                ro_number: i.3,
            });
        }
    }
    Ok(tensorvec)
}

pub fn predictiondata(pathfile: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut record: Vec<Vec<f64>> = Vec::new();
    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);
    for file in fileread.lines() {
        let fileline = file.expect("line not present");
        record.push(vec![fileline.parse::<f64>().unwrap()]);
    }
    Ok(record)
}

pub fn poplogisticclassification(
    pathfile: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdir_tensor(pathfile).unwrap();
    let mut vecgenotype: Vec<Vec<f64>> = Vec::new();
    let mut classlabels: Vec<i32> = Vec::new();
    for i in inputvariable.iter() {
        if i.quality_55 < quality.parse::<f64>().unwrap() {
            classlabels.push(0);
        } else if i.quality_55 > quality.parse::<f64>().unwrap() {
            classlabels.push(1);
        }
        vecgenotype.push(vec![i.gentoype_54, i.quality_55, i.ro_number]);
    }
    let featurevalue = DenseMatrix::from_2d_vec(&vecgenotype).unwrap();
    let model = LogisticRegression::fit(&featurevalue, &classlabels, Default::default()).unwrap();
    let prediction = predictiondata(genotypespath).unwrap();
    let finalvalupred = DenseMatrix::from_2d_vec(&prediction).unwrap();
    let predictions_value = model.predict(&finalvalupred).unwrap();
    let accuracypred = accuracy(&classlabels, &predictions_value);
    let mut filewrite = File::open("predictedvalue.txt").expect("The file not present");
    writeln!(
        filewrite,
        "The model predicted accuracy is: {}",
        accuracypred
    )
    .expect("file not found");
    for i in predictions_value.into_iter() {
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok("The logistic model has finished with the accuracy".to_string())
}

pub fn popknnclassification(
    pathfile: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdir_tensor(pathfile).unwrap();
    let mut vecgenotype: Vec<Vec<f64>> = Vec::new();
    let mut classlabels: Vec<i32> = Vec::new();
    for i in inputvariable.iter() {
        if i.quality_55 < quality.parse::<f64>().unwrap() {
            classlabels.push(0);
        } else if i.quality_55 > quality.parse::<f64>().unwrap() {
            classlabels.push(1);
        }
        vecgenotype.push(vec![i.gentoype_54, i.quality_55, i.ro_number]);
    }
    let featurevalue = DenseMatrix::from_2d_vec(&vecgenotype).unwrap();
    let model = KNNClassifier::fit(&featurevalue, &classlabels, Default::default()).unwrap();
    let prediction = predictiondata(genotypespath).unwrap();
    let finalvalupred = DenseMatrix::from_2d_vec(&prediction).unwrap();
    let predictions_value = model.predict(&finalvalupred).unwrap();
    let accuracypred = accuracy(&classlabels, &predictions_value);
    let mut filewrite = File::open("predictedvalue.txt").expect("The file not present");
    for i in predictions_value.into_iter() {
        writeln!(
            filewrite,
            "The model has predicted the value with the accuracy:{}",
            accuracypred
        )
        .expect("file not found");
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok("The knn classification model has finished with the accuracy".to_string())
}

pub fn poprandomforestclassification(
    pathfile: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdir_tensor(pathfile).unwrap();
    let mut vecgenotype: Vec<Vec<f64>> = Vec::new();
    let mut classlabels: Vec<i32> = Vec::new();
    for i in inputvariable.iter() {
        if i.quality_55 < quality.parse::<f64>().unwrap() {
            classlabels.push(0);
        } else if i.quality_55 > quality.parse::<f64>().unwrap() {
            classlabels.push(1);
        }
        vecgenotype.push(vec![i.gentoype_54, i.quality_55, i.ro_number]);
    }
    let featurevalue = DenseMatrix::from_2d_vec(&vecgenotype).unwrap();
    let model =
        RandomForestClassifier::fit(&featurevalue, &classlabels, Default::default()).unwrap();
    let prediction = predictiondata(genotypespath).unwrap();
    let finalvalupred = DenseMatrix::from_2d_vec(&prediction).unwrap();
    let predictions_value = model.predict(&finalvalupred).unwrap();
    let accuracypred = accuracy(&classlabels, &predictions_value);
    let mut filewrite = File::open("predictedvalue.txt").expect("The file not present");
    for i in predictions_value.into_iter() {
        writeln!(
            filewrite,
            "The model has predicted the value with the accuracy:{}",
            accuracypred
        )
        .expect("file not found");
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok("The random classification model has finished with the accuracy".to_string())
}

pub fn pop_logistic(
    pathfile: &str,
    variant: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdir_tensor(pathfile).unwrap();
    let mut vecgenotype: Vec<Vec<f64>> = Vec::new();
    let mut classlabels: Vec<i32> = Vec::new();
    for i in inputvariable.iter() {
        if i.nameallele == variant && i.quality_55 < quality.parse::<f64>().unwrap() {
            classlabels.push(0);
        } else if i.quality_55 > quality.parse::<f64>().unwrap() {
            classlabels.push(1);
        }
        vecgenotype.push(vec![i.gentoype_54, i.quality_55, i.ro_number]);
    }
    let featurevalue = DenseMatrix::from_2d_vec(&vecgenotype).unwrap();
    let model = LogisticRegression::fit(&featurevalue, &classlabels, Default::default()).unwrap();
    let prediction = predictiondata(genotypespath).unwrap();
    let finalvalupred = DenseMatrix::from_2d_vec(&prediction).unwrap();
    let predictions_value = model.predict(&finalvalupred).unwrap();
    let accuracypred = accuracy(&classlabels, &predictions_value);
    let mut filewrite = File::open("predictedvalue.txt").expect("The file not present");
    writeln!(
        filewrite,
        "The model predicted accuracy is: {}",
        accuracypred
    )
    .expect("file not found");
    for i in predictions_value.into_iter() {
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok("The logistic model has finished with the accuracy".to_string())
}

pub fn pop_knn(
    pathfile: &str,
    variant: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdir_tensor(pathfile).unwrap();
    let mut vecgenotype: Vec<Vec<f64>> = Vec::new();
    let mut classlabels: Vec<i32> = Vec::new();
    for i in inputvariable.iter() {
        if i.nameallele == variant && i.quality_55 < quality.parse::<f64>().unwrap() {
            classlabels.push(0);
        } else if i.quality_55 > quality.parse::<f64>().unwrap() {
            classlabels.push(1);
        }
        vecgenotype.push(vec![i.gentoype_54, i.quality_55, i.ro_number]);
    }
    let featurevalue = DenseMatrix::from_2d_vec(&vecgenotype).unwrap();
    let model = KNNClassifier::fit(&featurevalue, &classlabels, Default::default()).unwrap();
    let prediction = predictiondata(genotypespath).unwrap();
    let finalvalupred = DenseMatrix::from_2d_vec(&prediction).unwrap();
    let predictions_value = model.predict(&finalvalupred).unwrap();
    let accuracypred = accuracy(&classlabels, &predictions_value);
    let mut filewrite = File::open("predictedvalue.txt").expect("The file not present");
    writeln!(
        filewrite,
        "The model predicted accuracy is: {}",
        accuracypred
    )
    .expect("file not found");
    for i in predictions_value.into_iter() {
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok("The logistic model has finished with the accuracy".to_string())
}

pub fn pop_random(
    pathfile: &str,
    variant: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdir_tensor(pathfile).unwrap();
    let mut vecgenotype: Vec<Vec<f64>> = Vec::new();
    let mut classlabels: Vec<i32> = Vec::new();
    for i in inputvariable.iter() {
        if i.nameallele == variant && i.quality_55 < quality.parse::<f64>().unwrap() {
            classlabels.push(0);
        } else if i.quality_55 > quality.parse::<f64>().unwrap() {
            classlabels.push(1);
        }
        vecgenotype.push(vec![i.gentoype_54, i.quality_55, i.ro_number]);
    }
    let featurevalue = DenseMatrix::from_2d_vec(&vecgenotype).unwrap();
    let model =
        RandomForestClassifier::fit(&featurevalue, &classlabels, Default::default()).unwrap();
    let prediction = predictiondata(genotypespath).unwrap();
    let finalvalupred = DenseMatrix::from_2d_vec(&prediction).unwrap();
    let predictions_value = model.predict(&finalvalupred).unwrap();
    let accuracypred = accuracy(&classlabels, &predictions_value);
    let mut filewrite = File::open("predictedvalue.txt").expect("The file not present");
    writeln!(
        filewrite,
        "The model predicted accuracy is: {}",
        accuracypred
    )
    .expect("file not found");
    for i in predictions_value.into_iter() {
        writeln!(filewrite, "Value:\t{}", i).expect("line not present");
    }
    Ok("The logistic model has finished with the accuracy".to_string())
}
