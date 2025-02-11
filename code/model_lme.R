# This script performs a significance test, extracts bias scores (coefficient of the gender variable), and calculates effect size (R-squared).
# Reported bias score: gender coefficient (L224), significance test: p-value (L224), effect size: R2 (L209)

setwd("/path/to/code/")

#!/usr/bin/env Rscript
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

# Setting seeds 
set.seed(42)

option_list = list(
  make_option(c("-i", "--input_file"), type="character", 
              help="input file name", metavar="character"),
  make_option(c("-o", "--output_txt_file"), type="character",
              help="input file name", metavar="character"),
  make_option(c("-O", "--output_tsv_file"), type="character",
              help="input file name", metavar="character"),
  make_option(c("-s", "--selection"), type="character",
              help="selection criteria e.g., 2 indirect templates", metavar="character"),
  make_option(c("-f", "--factor"), type="character",
              help="factor1-4", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

workers <- availableWorkers()
cat(sprintf("#workders/#availableCores/#totalCores: %d/%d/%d, workers:\n", length(workers), availableCores(), detectCores()))

# workers can now be set to multiple cores to parallel 
plan(multisession, workers = availableCores())

##############################################################################
## significance test considering template_id, trait and gender.

dataset <- read.csv(opt$input_file, sep='\t', header=TRUE)

if (opt$selection =="t1_t2"){ # 2 indirect
  dataset <- dataset[dataset$template_id %in% c("template_1","template_2"),] 
} else if (opt$selection =="t3_t4"){ # 2 direct
  dataset <- dataset[dataset$template_id %in% c("template_3","template_4"),] 
} else if (opt$selection =="t1_to_t4"){ # 2 indirect + 2 direct
  dataset <- dataset[dataset$template_id %in% c("template_1","template_2","template_3","template_4"),] 
} else if (opt$selection =="t3_to_t6"){ # 4 direct
  dataset <- dataset[dataset$template_id %in% c("template_3","template_4","template_5","template_6"),] 
} else if (opt$selection =="t1_to_t6"){ # all
  dataset <- dataset
}

weight_vector <- 1/dataset$ppl_score
  
output <- lmer(association_score~gender + (1|template_id) + (1|trait), data = dataset, weights = weight_vector)

sink(file = opt$output_txt_file)

get_variance(output, c("all", "fixed", "random", "residual", "distribution", "dispersion",
                       "intercept", "slope", "rho01", "rho00"),verbose = TRUE)
print(strrep("-", 50))
print("")

fixed_effect_varaince <- get_variance_fixed(output,verbose = FALSE)

r2 <- r2_nakagawa(output) # marginal and conditional R2
print(r2)
print(strrep("-", 50))
print("")

# r2(output) # using performance(); same as r2_nakagawa

print(ranef(output))
print(strrep("-", 50))
print("")

print(fixef(output))
print(strrep("*", 50))
print("")

print(summary(output))
print(strrep("-", 50))
print("")

gender_comparison <- emmeans(output, pairwise ~ gender)
print(gender_comparison)
print(strrep("-", 50))
print("")

print(anova(output))

print(strrep("=", 50))
print("")

R2 <- partR2(output, partvars = c("gender"), R2_type = "marginal", 
             nboot = 1000, 
             parallel = TRUE)

print(summary (R2))

print(R2$boot_warnings)

print(strrep("-", 50))
print("")

###################################################
# Extract variance components
var_components <- VarCorr(output)

# Display the variance components
print(var_components)
print(strrep("*", 50))

summary_output <- summary(output)

gender_variance <- fixed_effect_varaince

# random effect variance
template_id_variance <- attr(var_components$template_id, "stddev")^2
trait_variance <- attr(var_components$trait, "stddev")^2
# residual_variance <- sigma(output)^2
residual_variance <- attr(var_components, "sc")^2

cat("gender_variance:", gender_variance, "\n")
cat("template_id_variance:", template_id_variance, "\n")
cat("trait_variance:", trait_variance, "\n")
cat("residual_variance:", residual_variance, "\n")
print(strrep("*", 50))

total_variance <- gender_variance + template_id_variance + trait_variance + residual_variance

gender_prop <- gender_variance/total_variance
template_id_prop <- template_id_variance / total_variance
trait_prop <- trait_variance / total_variance
residual_prop <- residual_variance/ total_variance

cat("Fixed effect R2:", gender_prop, "\n")
cat("template R2:", template_id_prop, "\n")
cat("trait R2:", trait_prop, "\n")
cat("residual R2:", residual_prop, "\n")
print(strrep("*", 50))

############################################################################################################################
#               Convert lmer output to dataframe and extract random effects varainces
############################################################################################################################
print(strrep("#", 50))
lmer_estimates <- as.data.frame(tidy(output))
print(lmer_estimates)

trait_std <- lmer_estimates[lmer_estimates$group=='trait' & lmer_estimates$term=='sd__(Intercept)',]$estimate
template_id_std <- lmer_estimates[lmer_estimates$group=='template_id' & lmer_estimates$term=='sd__(Intercept)',]$estimate


############################################################################################################################
#               Convert anova output to dataframe
############################################################################################################################
anova_sum_sq_estimates <- as.data.frame(tidy(anova(output)))
print(anova_sum_sq_estimates)

############################################################################################################################
#               Extract fixed effects coefficient, p-value and sum_of_square
############################################################################################################################

f1 <- c(lmer_estimates[lmer_estimates$term == "gendermale", ]$estimate,
        lmer_estimates[lmer_estimates$term == "gendermale", ]$p.value,
        anova_sum_sq_estimates[anova_sum_sq_estimates$term =='gender',]$meansq)

############################################################################################################################
#               Extract R2 and confidence interval for fixed and random effects
############################################################################################################################

R2_fix_effects_estimates <- as.data.frame(R2$R2, sep="\t") # returns R-squared values for fixed effects
R2_fix_effects_estimates <- R2_fix_effects_estimates %>% select('term', 'estimate', 'CI_lower', 'CI_upper') # select 4 columns only
R2_fix_effects_estimates_ <- R2_fix_effects_estimates[R2_fix_effects_estimates$term == c("Full", "gender"),] #select only two rows from R2_estimates
print(R2_fix_effects_estimates_)

variables <- c("Model","gender","trait", "template_id", "Residual", "Fixed")

R2_mixed <- c(R2_fix_effects_estimates_$estimate, # R2 for [Model, gender]
              
              trait_prop, template_id_prop, residual_prop, gender_prop
)

CI_LB_mixed <- c(R2_fix_effects_estimates_$CI_lower, # CI_lower_bound for [Model, gender]
                 -1, -1, -1, -1 # store -1 as we do not know confidence interval for trait, template_id, residual. Also, store -1 for fixed effect
)

CI_UB_mixed <- c(R2_fix_effects_estimates_$CI_upper, # CI_upper_bound for [Model, gender],
                 -1, -1, -1, -1 # store -1 as we do not know confidence interval for trait, template_id, residual. Also, store -1 for fixed effect
)

df <- data.frame(variables, R2_mixed, CI_LB_mixed, CI_UB_mixed)

# rename columns names
colnames(df)[which(names(df) == "R2_mixed")] <- "R2"
colnames(df)[which(names(df) == "CI_LB_mixed")] <- "LB"
colnames(df)[which(names(df) == "CI_UB_mixed")] <- "UB"

# add extra information about selection, LM, RM , factor, sentiment, 
df$selection <- c(rep(opt$selection, nrow(df)))
df$factor <- c(rep(opt$factor, nrow(df)))

############################################################################################################################
#                       Print required information
############################################################################################################################
print("gender coefficient, p-value, sum_sq")
print(f1)

print("trait std")
print(trait_std)

print("template_id_std")
print(template_id_std)

# add extra information gender_coefficient, ppl_coefficient, p-value, sum_sq, trait std, template_type_id std with all emtpy
df$gender_coef <- c(rep("NA", nrow(df)))
df$ppl_coef <- c(rep("NA", nrow(df)))
df$p_value <- c(rep("NA", nrow(df)))
df$sum_sq <- c(rep("NA", nrow(df)))
df$persona_std <- c(rep("NA", nrow(df)))
df$template_std <- c(rep("NA", nrow(df)))

# update gender_coefficient, p-value and sum_sq
df <- within(df, {
  f <- variables == 'gender'
  gender_coef[f] <- f1[1]
  p_value[f] <- f1[2]
  sum_sq[f] <- f1[3]
})

# update trait std
df <- within(df, {
  f <- variables == 'trait'
  persona_std[f] <-trait_std
})

# update template std
df <- within(df, {
  f <- variables == 'template_id'
  template_std[f] <-template_id_std
})

# remove extra column named "f"
drop <- c("f")
df <- df[,!(names(df) %in% drop)]

print(df,row.names = FALSE)

# save dataframe to tsv file
write.table(df, file=opt$output_tsv_file, quote=FALSE, row.names = FALSE, col.names = TRUE,sep='\t')

sink(file = NULL)
