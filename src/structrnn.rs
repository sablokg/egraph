#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct RefTrain {
    pub refallele: String,
    pub id: String,
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct AltTrain {
    pub altallele: String,
    pub id: String,
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct FunctionTrain {
    pub functionalt: String,
    pub id: String,
}
