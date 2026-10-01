# egraph

- Machine learning crate to how to use the population variant data from the eVai predictions or the other variants for the machine learning and predicts and confirm where the variant data is not annotated with respect to the previous population. 
- See the test files as how to prepare the data for the variant classification.
- a complete classification on the variants coming from the population so that you can confirm any unanotated variant or even check.
- This allows for the classification of the variants where a population has been studied and you want to implicit a classification model. 

### New Feature
- Added Autoencoder Neural Network. 

```
cargo build
```

```
          ____   ____       _      ____    _   _ 
   ___   / ___| |  _ \     / \    |  _ \  | | | |
  / _ \ | |  _  | |_) |   / _ \   | |_) | | |_| |
 |  __/ | |_| | |  _ <   / ___ \  |  __/  |  _  |
  \___|  \____| |_| \_\ /_/   \_\ |_|     |_| |_|
                                                 

classification for variant annotation.
       ************************************************
       Author Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************

Usage: egraph <COMMAND>

Commands:
  classifier          Classify a single population
  variant-classifier  filter and classify the variants on the specific variant types for a single population
  population          Classify on an entire population
  population-variant  Classify an entire population over a specific variant
  ref-allele          Ref Alt Neural Network
  alt-allele          Alt Alt Neural Network
  gene-effect         ref Alt Neural Network
  help                Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version

```

