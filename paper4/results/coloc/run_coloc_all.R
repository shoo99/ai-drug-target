#!/usr/bin/env Rscript
# Systematic COLOC for all TWAS-significant genes (BrainMeta cis-eQTL x PGC MDD2025)
suppressMessages({library(coloc); library(data.table)})
WS <- "/mgmt-data/gwas-mdd"; DIR <- file.path(WS,"results/coloc/all")
S <- 0.256491; NCC <- 341197 + 989052
g <- fread(file.path(WS,"data/gwas/pgc_clean.tsv"), select=c("ID","BETA","SE","FCAS"))
setnames(g, c("ID","BETA","SE","FCAS"), c("snp","b_gwas","se_gwas","maf_gwas"))
g <- g[!is.na(b_gwas) & se_gwas>0 & maf_gwas>0 & maf_gwas<1 & !duplicated(snp)]
files <- list.files(DIR, pattern="_eqtl\\.txt$", full.names=TRUE)
res <- data.frame()
for (f in files) {
  gene <- sub("_eqtl.txt$","",basename(f))
  e <- tryCatch(fread(f, select=c("SNP","Freq","b","SE")), error=function(x)NULL)
  if (is.null(e) || nrow(e)<50) next
  setnames(e, c("SNP","Freq","b","SE"), c("snp","maf_eqtl","b_eqtl","se_eqtl"))
  e <- e[!is.na(b_eqtl) & se_eqtl>0 & maf_eqtl>0 & maf_eqtl<1 & !duplicated(snp)]
  e[maf_eqtl>0.5, maf_eqtl := 1-maf_eqtl]
  m <- merge(g, e, by="snp")
  if (nrow(m) < 50) next
  D1 <- list(beta=m$b_gwas, varbeta=m$se_gwas^2, snp=m$snp, type="cc", s=S, N=NCC, MAF=m$maf_gwas)
  D2 <- list(beta=m$b_eqtl, varbeta=m$se_eqtl^2, snp=m$snp, type="quant", N=2865, MAF=m$maf_eqtl)
  r <- tryCatch(suppressWarnings(coloc.abf(D1,D2)), error=function(x)NULL)
  if (is.null(r)) next
  pp <- r$summary
  res <- rbind(res, data.frame(gene=gene, nsnp=nrow(m),
        PP3=round(pp["PP.H3.abf"],4), PP4=round(pp["PP.H4.abf"],4)))
}
res <- res[order(-res$PP4),]
write.csv(res, file.path(WS,"results/coloc/coloc_all_summary.csv"), row.names=FALSE)
cat("genes tested:", nrow(res), "\n")
cat("PP4>0.8 (colocalized):", sum(res$PP4>0.8), "\n")
cat("PP4 0.5-0.8 (suggestive):", sum(res$PP4>0.5 & res$PP4<=0.8), "\n")
cat("\nTop 15 colocalized:\n"); print(head(res[res$PP4>0.5,], 15))
