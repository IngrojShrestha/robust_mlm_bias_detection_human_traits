#!/usr/bin/env Rscript
setwd("path/to/code/crowspairs/code/")
library("optparse")
library(performance)
library(modelbased)
library(broom.mixed)
library(psycho)
library(gmodels)
library(dplyr)
library(data.table)
library(lmerTest)
library(lme4)
library(report)
library(partR2)
library(parallel) # for using parallel::mclapply() and checking #totalCores on compute nodes / workstation: detectCores()
library(future) # for checking #availble cores / workers on compute nodes / workstation: availableWorkers() / availableCores() 
library(nlme)
library(insight) # for get_variance
library(emmeans)
library(rptR)

options(max.print=1000000)

option_list = list(
  make_option(c("-i", "--input_file"), type="character", help="input file name", metavar="character")
  ); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

workers <- availableWorkers()
plan(multisession, workers = availableCores())

cat(sprintf("#workders/#availableCores/#totalCores: %d/%d/%d, workers:\n", length(workers), availableCores(), detectCores()))

dataset <- read.csv(opt$input_file, sep='\t', header=TRUE)

# View(dataset)

weight_vector <- 1/dataset$perplexity

output <- lmer(score~category + (1|sent_length_cat), data = dataset, weights = weight_vector)

# get_variance(output, c("all", "fixed", "random", "residual", "distribution", "dispersion",
#                        "intercept", "slope", "rho01", "rho00"),verbose = TRUE)

# print(strrep("-", 50))
# print("")

# fixed_effect_varaince <- get_variance_fixed(output,verbose = FALSE)

# r2 <- r2_nakagawa(output) # marginal and conditional R2
# print(r2)
# print(strrep("-", 50))
# print("")

# print(ranef(output))
# print(strrep("-", 50))
# print("")

# print(fixef(output))
# print(strrep("*", 50))
# print("")

# Bias Score given by coefficient of 'categorystereo' [bias score = sterotypical PLLScore - anti-stereotypical PLLScore]
print(summary(output))
print(strrep("-", 50))
print("")

# This can also be used to get bias score
gender_comparison <- emmeans(output, pairwise ~ category)
print(gender_comparison)
print(strrep("-", 50))
print("")

# print(anova(output))

print(strrep("=", 50))
print("")

R2 <- partR2(output, partvars = c("category"), R2_type = "marginal", 
             nboot = 1000, 
             parallel = TRUE)

# Effect size = R2 (marginal) 
# Also provides Confidence Interval for R2
print(summary (R2))