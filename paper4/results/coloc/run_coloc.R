#!/usr/bin/env Rscript
# COLOC (coloc.abf) for priority MDD genes: GWAS (PGC MDD2025) x BrainMeta cis-eQTL
suppressMessages({library(coloc); library(data.table)})
WS <- "/mgmt-data/gwas-mdd"
S  <- 0.256491          # case fraction
NCC <- 341197 + 989052  # total N for cc

# GWAS once
g <- fread(file.path(WS,"data/gwas/pgc_clean.tsv"),
           select=c("ID","EA","NEA","BETA","SE","FCAS"))
setnames(g, c("ID","BETA","SE","FCAS"), c("snp","b_gwas","se_gwas","maf_gwas"))
g <- g[!is.na(b_gwas) & se_gwas>0 & !duplicated(snp)]

genes <- c("DRD2","NEGR1","SLC12A5","GPX1","FURIN","DCC")
res <- data.frame()
for (gn in genes) {
  f <- file.path(WS,"results/coloc",paste0(gn,"_eqtl.txt"))
  if (!file.exists(f)) next
  e <- fread(f, select=c("SNP","Freq","b","SE"))
  setnames(e, c("SNP","Freq","b","SE"), c("snp","maf_eqtl","b_eqtl","se_eqtl"))
  e <- e[!is.na(b_eqtl) & se_eqtl>0 & maf_eqtl>0 & maf_eqtl<1 & !duplicated(snp)]
  e[maf_eqtl>0.5, maf_eqtl := 1-maf_eqtl]
  m <- merge(g, e, by="snp")
  m <- m[maf_gwas>0 & maf_gwas<1]
  if (nrow(m) < 50) { cat(gn,": too few shared SNPs (",nrow(m),")\n"); next }
  D1 <- list(beta=m$b_gwas, varbeta=m$se_gwas^2, snp=m$snp, type="cc", s=S, N=NCC, MAF=m$maf_gwas)
  D2 <- list(beta=m$b_eqtl, varbeta=m$se_eqtl^2, snp=m$snp, type="quant", N=2865, MAF=m$maf_eqtl)
  r <- tryCatch(coloc.abf(D1,D2), error=function(x){cat(gn,"ERR",conditionMessage(x),"\n");NULL})
  if (is.null(r)) next
  pp <- r$summary
  res <- rbind(res, data.frame(gene=gn, nsnp=nrow(m),
            PP0=pp["PP.H0.abf"], PP1=pp["PP.H1.abf"], PP2=pp["PP.H2.abf"],
            PP3=pp["PP.H3.abf"], PP4=pp["PP.H4.abf"]))
  cat(sprintf("%-9s nsnp=%d  PP4(coloc)=%.3f  PP3(distinct)=%.3f\n", gn, nrow(m), pp["PP.H4.abf"], pp["PP.H3.abf"]))
}
write.csv(res, file.path(WS,"results/coloc/coloc_summary.csv"), row.names=FALSE)
cat("\n=== saved results/coloc/coloc_summary.csv ===\n")
