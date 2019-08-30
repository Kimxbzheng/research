library(GEOquery)

# SOFT file
gds858 <- getGEO('GDS858', destdir = './GEOdata')

# chip information
gp196 <- gpl96 <- getGEO('GPL96', destdir='./GEOdata') 

# GSE
gse1009 <- getGEO('GSE1009', destdir='./GEOdata')

Table(gds858)
names(Meta(gds858))
eset <- GDS2eSet(gds858, do.log2=TRUE)

# GSE
gse107011 <- getGEO('GSE107011', destdir='./GEOdata', GSEMatrix=TRUE)

# getGEO(GEO = NULL, filename = NULL, destdir = tempdir(), GSElimits=NULL, GSEMatrix=TRUE,AnnotGPL=FALSE)

expr_dat107019=read.table("./GEOdata/GSE107019-GPL10558_series_matrix.txt",comment.char="!",stringsAsFactors=F)

exprSet = exprs(gse107019[[1]])
pdata = pData(gse107019[[1]])
