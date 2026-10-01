use rand::SeedableRng;
use rand::rngs::StdRng;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::accuracy;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
- a streamline machine learning crate to how to use the population variant
  data from the eVai or the other variants for the machine learning and predicts
  and confirm where the variant data is not annotated.
  see the test files as how to prepare the data for the vairant classification.
*/

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct AutoencoderData {
    pub nameallele: String,
    pub gentoype_54: f64,
    pub quality_55: f64,
    pub ro_number: f64,
}

/*
Taking the variant and making the classification pass and
here i have all the variants that have been annotated and passed.
*/

type EGRAPHTYPE = Vec<AutoencoderData>;

pub fn readdata(path: &str) -> Result<EGRAPHTYPE, Box<dyn std::error::Error>> {
    let fileopen = File::open(path).expect("file not present");
    let fileread = BufReader::new(fileopen);
    let mut tensorvec: Vec<AutoencoderData> = Vec::new();
    for i in fileread.lines() {
        let line = i.expect("file not present");
        let linevec = line.split("\t").collect::<Vec<_>>();
        if linevec[56] == "pass" {
            tensorvec.push(AutoencoderData {
                nameallele: linevec[4].to_string(),
                gentoype_54: linevec[54].parse::<f64>().unwrap(),
                quality_55: linevec[55].parse::<f64>().unwrap(),
                ro_number: linevec[58].parse::<f64>().unwrap(),
            })
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


pub fn logisticclassification(
    pathfile: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdata(pathfile).unwrap();
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

pub fn knnclassification(
    pathfile: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdata(pathfile).unwrap();
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

pub fn randomforestclassification(
    pathfile: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdata(pathfile).unwrap();
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

/*
 filter off those variants and then put the feature selection on those variants only
*/

pub fn variantclasslabel_logistic(
    pathfile: &str,
    variant: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdata(pathfile).unwrap();
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

pub fn variantclasslabel_knn(
    pathfile: &str,
    variant: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdata(pathfile).unwrap();
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

pub fn variantclasslabel_random(
    pathfile: &str,
    variant: &str,
    quality: &str,
    genotypespath: &str,
) -> Result<String, Box<dyn Error>> {
    let _seed = StdRng::seed_from_u64(2811);
    let inputvariable = readdata(pathfile).unwrap();
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
