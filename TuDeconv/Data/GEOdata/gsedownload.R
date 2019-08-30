library(GEOquery)

downGSE <- function(studyID = "GSE1009", destdir = "./") {
  
  eSet <- getGEO(studyID, destdir = destdir, getGPL = F)
  
  exprSet = exprs(eSet[[1]])
  pdata = pData(eSet[[1]])
  
  write.csv(exprSet, paste0("./TD_Data/",studyID, "_exprSet.csv"))
  write.csv(pdata, paste0("./TD_Data/",studyID, "_metadata.csv"))
  return(eSet)
  
}

Data = read.csv(file="./GEOdata/genedata.csv", header=TRUE, sep=",")

xR = nrow(Data)

for (c in 1:xR)
{
  study = Data[c,1]
  downGSE(studyID = study, destdir = "./GEOdata/")
}
