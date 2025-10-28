#____________________________
#Predict drop out from climate variables. 
#____________________________




#____________________________
#### Notes ####
# Of note is that tidymodels can only use the caseweights and not the complex
  
# survey weights.
#____________________________





#_______________________________________________________________________________
####LOAD PACKAGES####
library(survey) # Only used for weighted survey analysis not the ML. 
library(tidymodels)
library(tidyverse)
library(ranger)      # For random forest engine
library(rpart)       # For decision tree engine
library(rpart.plot)  # For visualizing decision trees
library(themis)
library(forcats)
library(glmnet) #For LASSO
library(corrr)
library(skimr)
options(scipen = 999)

# Load R Data File
load("els_02_12_byf3pststu_v1_0.rdata") # The data file.
load("els_02_12_byf3stubrr_v1_0.rdata") # The weights file.
#_______________________________________________________________________________





#_______________________________________________________________________________
#####KEEP THESE VARIABLES####

# Create vector of selected variables

keepvars <- c(
   "STU_ID", # student id -- who this?
   "SCH_ID", # school id -- what school?
   "STRAT_ID", # stratum id
   "psu", # primary sampling unit - school level
   "F1SCH_ID", # school f1 id
   "F1UNIV1", # BY and F1 student status
   "F1UNIV2A",
   "F1UNIV2B",
   "F2UNIV1",
   "F2UNIV_P",
   "F3UNIV",
   "F3UNIVG10",
   "F3UNIVG12",
   "G10COHRT",
   "G12COHRT",
   "bystuwt",
   "byexpwt",
   "bysex", # gender
   "byrace", # race
   "F1QWT",
   "F1PNLWT",
   "F1EXPWT",
   "F1XPNLWT",
   "F1DOSTAT", # Target: Dropout (1= Dropout)
   "F1TRSCWT",
   "F2QTSCWT",
   "F2QWT",
   "F2F1WT",
   "F2BYWT",
   "F3QWT",
   "F3BYPNLWT",
   "F3F1PNLWT",
   "F3QTSCWT",
   "F3BYTSCWT",
   "F3F1TSCWT",
   "F3QTSCWT_O",
   "F3BYTSCWT_O",
   "F3F1TSCWT_O",
   "pswt",
   "F3BYPNLPSWT",
   "F3BYTSCPSWT",
   "F3F1PNLPSWT",
   "F3F1TSCPSWT",
   "F3QPSWT",
   "F3QTSCPSWT",
   "pstscwt",
   "BYS20A", # stds get along with teachers.
   "BYS20B", # there is real school spirit.
   "BYS20C", # students make diverse friends
   "BYS20D", # other students disrupt class
   "BYS20E", # teaching is good.
   "BYS20F", # teachers are interested in students.
   "BYS20G", # when i work hard, teachers praise.
   "BYS20H", # teachers put me down.
   "BYS20I", # students put me down.
   "BYS20J", # i dont feel safe.
   "BYS20K", # student disruptions get in the way.
   "BYS20L", # student misbehavior get in the way.
   "BYS20M", # there are gangs in school.
   "BYS20N", # racial fights often occur. 
   "BYS21A", # all know school rules.
   "BYS21B", # the rules are fair.
   "BYS21C", # punishment for break rules is same no matter who.
   "BYS21D", # school rules are strictly enforced.
   "BYS21E", # if rule broken, students know punishment kind to follow. 
   "BYS22A", # something stolen from me at school.
   "BYS22B", # Someone offered to sell me drugs at school
   "BYS22C", # someone threatened to hurt me at school
   "BYS22D", # got in a fight
   "BYS22E", # hit me
   "BYS22F", # force to take things from me
   "BYS22G", # damaged belonging
   "BYS22H", # bullied me
   "BYS27H"  # teachers expect me to exceed
)
#_______________________________________________________________________________








#_______________________________________________________________________________
#####SELECT VARIABLES AND MERGE DATA FILE w. WEIGHTS####

# Create new object containing only selected variables
els_data_kept_variables<- els_02_12_byf3pststu_v1_0[keepvars]


# Merge with BRR weights file
els_data_with_brr <- merge(els_data_kept_variables, 
                           els_02_12_byf3stubrr_v1_0, 
                           by = "STU_ID")
#_______________________________________________________________________________














#_______________________________________________________________________________
##### CONVERT F1DOSTAT TO A FACTOR AND MAKE BINARY####

# Create binary dropout variable
els_data_with_brr <- els_data_with_brr %>%
  mutate(
    dropout_binary = case_when(
      F1DOSTAT == 0 ~ 0,  # Not dropout
      F1DOSTAT == 1 ~ 1,  # Dropout
      F1DOSTAT == 2 ~ 0,  # Alternative completer -> Not dropout
      F1DOSTAT == 3 ~ 1,  # Student/school report dropout -> Dropout
      F1DOSTAT == 4 ~ NA_real_,  # Out of scope -> Exclude
      TRUE ~ NA_real_ #"anything i didnt plan, for count as missing"
    ),
    dropout_factor = factor(dropout_binary, 
                            levels = c(0, 1), 
                            labels = c("Not_Dropout", "Dropout"))
  )


# Relevel here so "Dropout" is treated as the positive event

els_data_with_brr <- els_data_with_brr %>%
  mutate(dropout_factor = fct_relevel(dropout_factor, "Dropout", "Not_Dropout"))






#____________________________
####Scaled Weights####
# Created scaled weights in your data preparation step (so models can converge)
# 
els_data_with_brr <- els_data_with_brr %>%
  mutate(
    
# case_wts =importance_weights(bystuwt.x) = takes the original survey weights.
#wraps them in importance_weights() telling tidmodels these are case weights
    #importance_weights() is a tidymodels function.
    #.x was appended to newly created variables on merger.
    
    case_wts = importance_weights(bystuwt.x),

    
#create scaled survey weights by divididng each weight by the average weight.
  # This normalizes the weights around 1.0
    case_wts_scaled = importance_weights(bystuwt.x / mean(bystuwt.x, na.rm = TRUE))
  )


# Then round them so the model converges (expects whole numbers)
# Use pmax(1, ...) to ensure no weights are zero (students wouldnt contribute)
els_data_with_brr <- els_data_with_brr %>%
  mutate(
    case_wts_rounded = importance_weights(pmax(1, round(bystuwt.x / mean(bystuwt.x, na.rm = TRUE))))
  )


#keep only the variables needed for ML
els_data_ml <- els_data_with_brr %>%
  select(STU_ID, SCH_ID, STRAT_ID, psu, 
         F1DOSTAT, dropout_factor, dropout_binary,
         bystuwt.x, case_wts, case_wts_scaled, case_wts_rounded,
         bysex, byrace,
         BYS20A, BYS20B, BYS20C, BYS20D, BYS20E, BYS20F, BYS20G, 
         BYS20H, BYS20I, BYS20J, BYS20K, BYS20L, BYS20M, BYS20N,
         BYS21A, BYS21B, BYS21C, BYS21D, BYS21E,
         BYS22A, BYS22B, BYS22C, BYS22D, BYS22E, BYS22F, BYS22G, BYS22H,
         BYS27H)
#_______________________________________________________________________________








#_______________________________________________________________________________


#_______________________________________________________________________________
#####MISSING####
# Replace - values with missing (NA)
# Replace negative values with missing (NA) for ALL attitude/climate variables
els_data_ml <- els_data_ml %>%
  mutate(
    # BYS20 series (school climate/attitude)
    BYS20A = ifelse(BYS20A < 0, NA, BYS20A),
    BYS20B = ifelse(BYS20B < 0, NA, BYS20B),
    BYS20C = ifelse(BYS20C < 0, NA, BYS20C),
    BYS20D = ifelse(BYS20D < 0, NA, BYS20D),
    BYS20E = ifelse(BYS20E < 0, NA, BYS20E),
    BYS20F = ifelse(BYS20F < 0, NA, BYS20F),
    BYS20G = ifelse(BYS20G < 0, NA, BYS20G),
    BYS20H = ifelse(BYS20H < 0, NA, BYS20H),
    BYS20I = ifelse(BYS20I < 0, NA, BYS20I),
    BYS20J = ifelse(BYS20J < 0, NA, BYS20J),
    BYS20K = ifelse(BYS20K < 0, NA, BYS20K),
    BYS20L = ifelse(BYS20L < 0, NA, BYS20L),
    BYS20M = ifelse(BYS20M < 0, NA, BYS20M),
    BYS20N = ifelse(BYS20N < 0, NA, BYS20N),
    
    # BYS21 series (school rules/discipline)
    BYS21A = ifelse(BYS21A < 0, NA, BYS21A),
    BYS21B = ifelse(BYS21B < 0, NA, BYS21B),
    BYS21C = ifelse(BYS21C < 0, NA, BYS21C),
    BYS21D = ifelse(BYS21D < 0, NA, BYS21D),
    BYS21E = ifelse(BYS21E < 0, NA, BYS21E),
    
    # BYS22 series (school safety/victimization)
    BYS22A = ifelse(BYS22A < 0, NA, BYS22A),
    BYS22B = ifelse(BYS22B < 0, NA, BYS22B),
    BYS22C = ifelse(BYS22C < 0, NA, BYS22C),
    BYS22D = ifelse(BYS22D < 0, NA, BYS22D),
    BYS22E = ifelse(BYS22E < 0, NA, BYS22E),
    BYS22F = ifelse(BYS22F < 0, NA, BYS22F),
    BYS22G = ifelse(BYS22G < 0, NA, BYS22G),
    BYS22H = ifelse(BYS22H < 0, NA, BYS22H),
    
    # BYS27 series (teacher expectations)
    BYS27H = ifelse(BYS27H < 0, NA, BYS27H),
    
    # Demographics
    bysex = ifelse(bysex < 0, NA, bysex),
    byrace = ifelse(byrace < 0, NA, byrace)
  )








#____________________________
# Count missing values in each variable
els_data_ml %>%
  select(dropout_factor, 
         BYS20A, BYS20B, BYS20C, BYS20D, BYS20E, BYS20F, BYS20G, 
         BYS20H, BYS20I, BYS20J, BYS20K, BYS20L, BYS20M, BYS20N,
         BYS21A, BYS21B, BYS21C, BYS21D, BYS21E,
         BYS22A, BYS22B, BYS22C, BYS22D, BYS22E, BYS22F, BYS22G, BYS22H,
         BYS27H, bysex, byrace) %>%
  summarise(across(everything(), ~sum(is.na(.))))







#____________________________
# Percentage missing for each variable  
els_data_ml %>%
  select(dropout_factor, 
         BYS20A, BYS20B, BYS20C, BYS20D, BYS20E, BYS20F, BYS20G, 
         BYS20H, BYS20I, BYS20J, BYS20K, BYS20L, BYS20M, BYS20N,
         BYS21A, BYS21B, BYS21C, BYS21D, BYS21E,
         BYS22A, BYS22B, BYS22C, BYS22D, BYS22E, BYS22F, BYS22G, BYS22H,
         BYS27H, bysex, byrace, case_wts_rounded) %>%
  summarise(across(everything(), ~round(100 * mean(is.na(.)), 2)))
#____________________________







#____________________________
# Remove observations with missing values on any of the key variables
els_data_ml <- els_data_ml %>%
  select(STU_ID, SCH_ID, STRAT_ID, psu,  # Keep ID variables
         F1DOSTAT, dropout_factor, dropout_binary,
         bystuwt.x, case_wts, case_wts_scaled, case_wts_rounded,  # Keep weights
         bysex, byrace,  # Demographics
         BYS20A, BYS20B, BYS20C, BYS20D, BYS20E, BYS20F, BYS20G, 
         BYS20H, BYS20I, BYS20J, BYS20K, BYS20L, BYS20M, BYS20N,
         BYS21A, BYS21B, BYS21C, BYS21D, BYS21E,
         BYS22A, BYS22B, BYS22C, BYS22D, BYS22E, BYS22F, BYS22G, BYS22H,
         BYS27H) %>%
  drop_na()  # Remove any row with missing values







#____________________________
# Check missing values (should all be 0 now)
els_data_ml %>%
  summarise(across(everything(), ~sum(is.na(.))))




#_______________________________________________________________________________
#####LABELS####




#____________________________
#Add labels - bysex

# Add labels to bysex (1 = Male, 2 = Female)

els_data_ml <- els_data_ml %>%
  mutate(
    bysex = factor(bysex, levels = c(1, 2), labels = c("Male", "Female"))
  )

#check
table(els_data_ml$bysex, useNA = "always")
#____________________________








#____________________________
# Add labels - byrace #

els_data_ml <- els_data_ml %>%
  mutate(
    byrace = factor(byrace, 
                    levels = c(1, 2, 3, 4, 5, 6, 7), 
                    labels = c("Am_Indian_Native", 
                               "Asian_PI", 
                               "Black", 
                               "Hispanic_NoRace", 
                               "Hispanic_Race", 
                               "Multiracial", 
                               "White"))
  )


#verify
table(els_data_ml$byrace, useNA = "always")
#____________________________









#_______________________________________________________________________________
#LABEL LIKERT AND FREQUENCY VARIABLES

# BYS20 and BYS21 series use Likert agreement scale (1-4)
likert_vars <- c(
  # BYS20 series (school climate/attitude)
  "BYS20A", "BYS20B", "BYS20C", "BYS20D", "BYS20E", "BYS20F", "BYS20G",
  "BYS20H", "BYS20I", "BYS20J", "BYS20K", "BYS20L", "BYS20M", "BYS20N",
  
  # BYS21 series (school rules/discipline)
  "BYS21A", "BYS21B", "BYS21C", "BYS21D", "BYS21E",
  
  # BYS27 series (teacher expectations)
  "BYS27H"
)

# BYS22 series use frequency scale (1-3)
frequency_vars <- c(
  "BYS22A", "BYS22B", "BYS22C", "BYS22D", "BYS22E", "BYS22F", "BYS22G", "BYS22H"
)

# Define response scales
likert_levels <- c("Strongly_Agree", "Agree", "Disagree", "Strongly_Disagree")
frequency_levels <- c("Never", "Once_or_Twice", "More_than_Twice")

# ____________________________
# Recode variables by their appropriate scale
els_data_ml <- els_data_ml |>
  mutate(
    # Likert scale variables (1-4)
    across(
      all_of(likert_vars),
      ~ factor(
        .x, # .x is a placeholder pronoun in the across function. specifies the current column
        levels = 1:4,
        labels = likert_levels,
        ordered = TRUE
      )
    ),
    # Frequency scale variables (1-3)
    across(
      all_of(frequency_vars),
      ~ factor(
        .x,
        levels = 1:3,
        labels = frequency_levels,
        ordered = TRUE
      )
    )
  )




# ____________________________
# Check counts (including NAs) per variable

els_data_ml %>%
  select(-case_wts, -case_wts_scaled, -case_wts_rounded) %>%
  summary()

#_______________________________________________________________________________





#_______________________________________________________________________________
####SAVE/LOAD CLEANED/MERGED DATA AFter Labels####
# --- Save merged dataset ---
saveRDS(els_data_ml, "els_data_ml.rds")

# --- Later: load it back without re-merging ---
els_data_ml <- readRDS("els_data_ml.rds")

# Quick check
glimpse(els_data_ml)
#_______________________________________________________________________________






#_______________________________________________________________________________
####CHECK ASSUMPTIONS ####

#____________________________
# CHECK MULTICOLLINEARITY / HIGH CORRELATIONS
#____________________________

# Select only numeric variables for correlation (updated variable list and dataset)
corr_data <- els_data_ml %>%
  select(BYS20A, BYS20B, BYS20C, BYS20D, BYS20E, BYS20F, BYS20G, 
         BYS20H, BYS20I, BYS20J, BYS20K, BYS20L, BYS20M, BYS20N,
         BYS21A, BYS21B, BYS21C, BYS21D, BYS21E,
         BYS22A, BYS22B, BYS22C, BYS22D, BYS22E, BYS22F, BYS22G, BYS22H,
         BYS27H) %>%
  mutate(across(everything(), as.numeric)) %>%   # convert factors back to numeric
  drop_na()                                      # drop missing

# Compute correlation matrix
# use = "pairwise.complete.obs") = use only obs that have complete variables
corr_matrix <- cor(corr_data, use = "pairwise.complete.obs")

# Find variable pairs with high correlation (> .8 in absolute value)
high_corr <- as.data.frame(as.table(corr_matrix)) %>%
  
  #Var1 != Var2: Remove self-correlations (BYS20A with BYS20A = 1.00)
  # abs(Freq) > 0.8: Keep only high correlations (positive OR negative above 0.8)
  filter(Var1 != Var2, abs(Freq) > 0.8) %>%
  arrange(desc(abs(Freq)))

print("High correlations (>0.8):")
print(high_corr)

# Correlation visualization

corr_data %>%
  correlate() %>%
  rearrange() %>%
  shave() %>%
  rplot()
#Result: No correlation for predictor variables above .08





#____________________________
# 1. CLASS BALANCE - Check target variable distribution
table(els_data_ml$dropout_factor, useNA = "always")
prop.table(table(els_data_ml$dropout_factor, useNA = "always")) * 100







#____________________________
# 2. MISSING DATA PATTERNS - How much missing data per variable?
missing_summary <- els_data_ml %>%
  select(dropout_factor, bysex, byrace, starts_with("BYS")) %>%
  
  #~sum(is.na(.)=
  # The tilde - creates a mini-function (formula) that will be applied to each column.
  # is.na() function - checks each value: missing=TRUE, not missing=FALSE
  # The dot - means "the current column being processed"
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_count") %>%
  mutate(missing_pct = round(missing_count / nrow(els_data_ml) * 100, 1)) %>%
  arrange(desc(missing_pct))

print (missing_summary, n =Inf)







#____________________________
# 3. SAMPLE SIZE - Do you have enough observations?
nrow(els_data_ml) # [1] 12405 total obs
nrow(els_data_ml %>% drop_na()) # [1] 12405 complete obs
sum(els_data_ml$dropout_factor == "Dropout", na.rm = TRUE) #733 dropout.
sum(els_data_ml$dropout_factor == "Not_Dropout", na.rm = TRUE) # 11,672 Non-dropout







#____________________________
# 4. PREDICTOR VARIABILITY - Are there predictors with little variation?
likert_vars <- c("BYS20A", "BYS20B", "BYS20C", "BYS20D", "BYS20E", "BYS20F", "BYS20G", 
                 "BYS20H", "BYS20I", "BYS20J", "BYS20K", "BYS20L", "BYS20M", "BYS20N",
                 "BYS21A", "BYS21B", "BYS21C", "BYS21D", "BYS21E",
                 "BYS22A", "BYS22B", "BYS22C", "BYS22D", "BYS22E", "BYS22F", "BYS22G", "BYS22H",
                 "BYS27H")





#____________________________
low_var_check <- els_data_ml %>%
  select(all_of(likert_vars)) %>%
  summarise(across(everything(), ~length(unique(.x[!is.na(.x)])))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "unique_values") %>%
  filter(unique_values <= 2) %>%
  arrange(unique_values)



low_var_check

#> Interpret: All your variables have adequate variation (3+ unique values)

#> low_var_check
#> A tibble: 0 × 2
#> ℹ 2 variables: variable <chr>, unique_values <int>




#____________________________
# More detailed variance check - look at response distributions
response_dist <- els_data_ml %>%
  select(all_of(likert_vars)) %>%
  map_dfr(~table(.x, useNA = "always") %>% as_tibble(), .id = "variable") %>%
  group_by(variable) %>%
  mutate(
    total = sum(n),
    prop = n / total,
    max_prop = max(prop)
  ) %>%
  slice_max(prop, n = 1) %>%
  filter(max_prop > 0.95)  # Flag if >95% in one category

print("Variables with >95% in one response category:")
print(response_dist)

#BYS22F (likely "Forced to give things") has 97.7% of students responding "Never" - 
# This is a near-zero variance variable.




#____________________________
#### Remove BYS22F variable/save/reload####
els_data_ml <- els_data_ml %>%
  select(-BYS22F)

# Save updated dataset
saveRDS(els_data_ml, "els_data_ml.rds")

# Load later
els_data_ml <- readRDS("els_data_ml.rds")





#____________________________
# 5. CATEGORICAL VARIABLE DISTRIBUTIONS
table(els_data_ml$bysex, useNA = "always")
table(els_data_ml$byrace, useNA = "always")





#____________________________
# 6. SURVEY WEIGHTS DISTRIBUTION
summary(as.numeric(els_data_ml$case_wts_rounded))







#____________________________
# 7. DATA TYPE VERIFICATION
class(els_data_ml$dropout_factor)
levels(els_data_ml$dropout_factor)
class(els_data_ml$case_wts_rounded)

#_______________________________________________________________________________











#_______________________________________________________________________________
##### PREDICTIVE SIGNAL ASSESSMENT - ALL VARIABLES ####
#_______________________________________________________________________________

# Load required libraries
library(pROC)
library(effectsize)
library(corrr)

#_______________________________________________________________________________
# 1. PREPARE DATA FOR SIGNAL ASSESSMENT
#_______________________________________________________________________________

# Select predictors and create binary outcome
predictors_for_signal <- els_data_ml %>%
  select(BYS20A, BYS20B, BYS20C, BYS20D, BYS20E, BYS20F, BYS20G, 
         BYS20H, BYS20I, BYS20J, BYS20K, BYS20L, BYS20M, BYS20N,
         BYS21A, BYS21B, BYS21C, BYS21D, BYS21E,
         BYS22A, BYS22B, BYS22C, BYS22D, BYS22E, BYS22G, BYS22H,
         BYS27H, bysex, byrace, dropout_factor) %>%
  drop_na()

# Create binary outcome (1 = Dropout, 0 = Not_Dropout)
predictors_for_signal <- predictors_for_signal %>%
  mutate(outcome_binary = as.numeric(dropout_factor == "Dropout"))

print(paste("Sample size for signal assessment:", nrow(predictors_for_signal)))
print(paste("Dropout rate:", round(mean(predictors_for_signal$outcome_binary) * 100, 1), "%"))

#_______________________________________________________________________________
# 2. CORRELATION ANALYSIS
#_______________________________________________________________________________
# Include this in the writeup - weak correlation with the outcome


# Convert ALL factors to numeric for correlation
predictors_numeric <- predictors_for_signal %>%
  mutate(
    # Convert all BYS variables from factors to numeric
    across(starts_with("BYS"), as.numeric),
    bysex_num = as.numeric(bysex),
    byrace_num = as.numeric(byrace)
  ) %>%
  select(-dropout_factor, -bysex, -byrace)

# Calculate correlations with outcome
correlations <- predictors_numeric %>%
  select(-outcome_binary) %>%
  map_dbl(~cor(.x, predictors_numeric$outcome_binary, use = "complete.obs"))

# Create correlation results table
correlation_results <- tibble(
  variable = names(correlations),
  correlation = correlations,
  abs_correlation = abs(correlations)
) %>%
  arrange(desc(abs_correlation))

print("=== CORRELATION WITH DROPOUT (Top 10) ===")
print(correlation_results, n = Inf)

# Correlation interpretation
# CORRELATION INTERPRETATION
  # Weak signal: |r| < 0.1
  # Small signal: |r| = 0.1-0.3
  # Medium signal: |r| = 0.3-0.5
  # Strong signal: |r| > 0.5

#_______________________________________________________________________________
# 3. UNIVARIATE AUC ANALYSIS
#_______________________________________________________________________________

# Calculate AUC for each predictor individually
univariate_aucs <- predictors_numeric %>%
  select(-outcome_binary) %>%
  map_dbl(~{
    tryCatch({
      roc_obj <- roc(predictors_numeric$outcome_binary, .x, direction = "<", quiet = TRUE)
      as.numeric(roc_obj$auc)
    }, error = function(e) NA_real_)
  })

# Create AUC results table
auc_results <- tibble(
  variable = names(univariate_aucs),
  auc = univariate_aucs,
  signal_strength = case_when(
    auc >= 0.7 ~ "Strong",
    auc >= 0.6 ~ "Moderate", 
    auc >= 0.55 ~ "Weak",
    auc < 0.55 ~ "Very Weak",
    TRUE ~ "Unable to calculate"
  )
) %>%
  arrange(desc(auc))

print("\n=== UNIVARIATE AUC ANALYSIS (Top 10) ===")
print(head(auc_results, 10))

# AUC interpretation
#AUC INTERPRETATION
#No signal: AUC ≈ 0.5
#Weak signal: AUC = 0.55-0.6
#Moderate signal: AUC = 0.6-0.7
#Strong signal: AUC > 0.7


#_______________________________________________________________________________







#_______________________________________________________________________________
##### SPLIT####
# We split the data and then address the class imbalance to prevent data leakage.
# We want to apply the imbalance fix only to the train data and leave the test 
  # data real world.

# Set seed for reproducibility
set.seed(123)

# Split data (stratified by outcome)
els_split <- initial_split(els_data_ml, prop = 0.8, strata = dropout_factor)
els_train <- training(els_split)
els_test <- testing(els_split)

#_______________________________________________________________________________







#_______________________________________________________________________________
##### TEST CLEANUP ####

# Drop rows with ANY missing values
els_test_clean <- els_test %>%
  drop_na()
#_______________________________________________________________________________








#_______________________________________________________________________________
#####UPSAMPLE####


# --- Clean training: drop rows with ANY missing values ---
els_train_clean <- els_train %>%
  drop_na()

# --- Check base counts ---
els_train_clean |> count(dropout_factor) |> mutate(prop = n / sum(n))



# UPSAMPLE training to 1:1 (keeps all majority rows; samples minority w/ replacement)
# (No leakage: only applied to training data)

# Counts by class
ctr <- els_train_clean |> count(dropout_factor) #counts the obs for each.
# Creates a summary table counting how many students are in each class
# Result: dropout_factor="Dropout" n=, dropout_factor="Not_Dropout" n=

maj_class <- ctr |> 
  slice_max(n, n = 1, with_ties = FALSE) |> 
# slice_max() finds the row with the highest count (the majority class)
# n = 1 means "give me only the top 1 row"
# with_ties = FALSE means "if there's a tie, just pick one"
  
  pull(dropout_factor) |>
  # pull() extracts just the dropout_factor column value from that row
  # This gives us the name of the majority class (e.g., "Not_Dropout")
  
  
  as.character()
# Converts the factor to character string for easier handling later
# Result: maj_class = "Not_Dropout"



n_maj<- ctr |> 
  slice_max(n, n = 1, with_ties = FALSE) |> 
# Again, finds the row with the highest count (same row as above)  
  
  
pull(n)
# This time, pull() extracts the COUNT (n) from that row
# This gives us the NUMBER of students in the majority class
# Result: n_maj = 12113 (or whatever the actual count is)


# Split train by class (dropping NA level if present)
spl <- split(els_train_clean, els_train_clean$dropout_factor, drop = TRUE)

#Splits the majority and non majority into two separate dfs.
maj_df <- spl[[maj_class]]
min_df <- bind_rows(spl[setdiff(names(spl), maj_class)]) 

# Sample minority up to majority size
min_df_up <- min_df |> slice_sample(n = n_maj, replace = TRUE)
# Takes 586 original Dropout students
# Samples 9,338 times WITH REPLACEMENT
# Result: 9,338 rows, with duplicates from the original data



# Bind balanced train
els_train_up <- bind_rows(maj_df, min_df_up) |> as_tibble()

# Verify balance
els_train_up |> count(dropout_factor)


# Skim the UPSAMPLED data
els_train_up |> 
  select(-case_wts, -case_wts_scaled, -case_wts_rounded) |>
  skim()


#____________________________
# Save upsampled training data
saveRDS(els_train_up, "els_train_up.rds")

# Load them back later
els_train_up <- readRDS("els_train_up.rds")

#_______________________________________________________________________________
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#_______________________________________________________________________________
#####SMOTE UPSAMPLING (Alternative to Manual Upsampling)####
#_______________________________________________________________________________
#_______________________________________________________________________________
#####SMOTE UPSAMPLING (Alternative to Manual Upsampling)####
#_______________________________________________________________________________

# --- Clean training: drop rows with ANY missing values ---
els_train_clean <- els_train %>%
  drop_na()

# --- Check base counts before SMOTE ---
cat("\n=== BEFORE SMOTE ===\n")
els_train_clean |> count(dropout_factor) |> mutate(prop = n / sum(n))

#____________________________
# Prepare data for SMOTE - convert factors to numeric
library(themis)

# Store factor levels for later reconstruction
bysex_levels <- levels(els_train_clean$bysex)
byrace_levels <- levels(els_train_clean$byrace)

# Store all the Likert variable names
likert_vars <- c("BYS20A", "BYS20B", "BYS20C", "BYS20D", "BYS20E", "BYS20F", "BYS20G", 
                 "BYS20H", "BYS20I", "BYS20J", "BYS20K", "BYS20L", "BYS20M", "BYS20N",
                 "BYS21A", "BYS21B", "BYS21C", "BYS21D", "BYS21E",
                 "BYS27H")

frequency_vars <- c("BYS22A", "BYS22B", "BYS22C", "BYS22D", "BYS22E", "BYS22G", "BYS22H")

# Calculate mean weight for synthetic minority class rows BEFORE SMOTE
minority_mean_weight <- els_train_clean %>%
  filter(dropout_factor == "Dropout") %>%
  summarise(mean_wt = mean(as.numeric(case_wts_rounded))) %>%
  pull(mean_wt)

original_n <- nrow(els_train_clean)

# Convert all factors to numeric for SMOTE, keep all other columns
els_train_for_smote <- els_train_clean %>%
  mutate(
    # Add marker to identify original rows
    .original_row = row_number(),
    # Convert ordered factors to numeric (preserves ordering)
    across(all_of(c(likert_vars, frequency_vars)), as.numeric),
    # Convert categorical factors to numeric
    bysex_num = as.numeric(bysex),
    byrace_num = as.numeric(byrace)
  ) %>%
  select(dropout_factor, .original_row, all_of(c(likert_vars, frequency_vars)), 
         bysex_num, byrace_num)

#____________________________
# Apply SMOTE
els_train_smote <- smote(
  els_train_for_smote,
  var = "dropout_factor",      # Target variable
  k = 10,                        # Number of nearest neighbors
  over_ratio = 1                # Upsample minority to match majority (1:1 ratio)
)

# Check balance after SMOTE
cat("\n=== AFTER SMOTE ===\n")
els_train_smote |> count(dropout_factor) |> mutate(prop = n / sum(n))

#____________________________
# Identify synthetic rows
els_train_smote <- els_train_smote %>%
  mutate(
    .smote_row = row_number(),
    # Rows beyond original_n are definitely synthetic
    is_synthetic = .smote_row > original_n
  )

#____________________________
# Convert numeric variables back to factors with proper labels

# Define response scales (same as in your original code)
likert_levels <- c("Strongly_Agree", "Agree", "Disagree", "Strongly_Disagree")
frequency_levels <- c("Never", "Once_or_Twice", "More_than_Twice")

# Convert back to factors
els_train_smote <- els_train_smote %>%
  mutate(
    # Convert Likert variables back to ordered factors
    across(
      all_of(likert_vars),
      ~ factor(
        round(.x),  # Round synthetic values to nearest integer
        levels = 1:4,
        labels = likert_levels,
        ordered = TRUE
      )
    ),
    # Convert frequency variables back to ordered factors
    across(
      all_of(frequency_vars),
      ~ factor(
        round(.x),  # Round synthetic values to nearest integer
        levels = 1:3,
        labels = frequency_levels,
        ordered = TRUE
      )
    ),
    # Convert bysex back to factor
    bysex = factor(
      round(bysex_num),
      levels = 1:2,
      labels = bysex_levels
    ),
    # Convert byrace back to factor
    byrace = factor(
      round(byrace_num),
      levels = 1:7,
      labels = byrace_levels
    )
  ) %>%
  select(-bysex_num, -byrace_num)

#____________________________
# Add back the other columns for original rows, create them for synthetic rows

els_train_up <- els_train_smote %>%
  left_join(
    els_train_clean %>% 
      select(STU_ID, SCH_ID, STRAT_ID, psu, F1DOSTAT, dropout_binary,
             bystuwt.x, case_wts, case_wts_scaled, case_wts_rounded) %>%
      mutate(.original_row = row_number()),
    by = ".original_row"
  ) %>%
  mutate(
    # Convert ID columns to character to allow combination with synthetic IDs
    STU_ID = as.character(STU_ID),
    SCH_ID = as.character(SCH_ID),
    STRAT_ID = as.character(STRAT_ID),
    psu = as.character(psu),
    
    # For synthetic rows, create new values
    STU_ID = if_else(is_synthetic, paste0("SYNTHETIC_", .smote_row), STU_ID),
    SCH_ID = if_else(is_synthetic, NA_character_, SCH_ID),
    STRAT_ID = if_else(is_synthetic, NA_character_, STRAT_ID),
    psu = if_else(is_synthetic, NA_character_, psu),
    F1DOSTAT = if_else(is_synthetic, 1, F1DOSTAT),  # Dropout
    dropout_binary = if_else(is_synthetic, 1, dropout_binary),
    bystuwt.x = if_else(is_synthetic, minority_mean_weight, bystuwt.x),
    case_wts = if_else(is_synthetic, 
                       importance_weights(minority_mean_weight), 
                       case_wts),
    case_wts_scaled = if_else(is_synthetic,
                              importance_weights(minority_mean_weight / mean(els_train_clean$bystuwt.x, na.rm = TRUE)),
                              case_wts_scaled),
    case_wts_rounded = if_else(is_synthetic,
                               importance_weights(pmax(1, round(minority_mean_weight / mean(els_train_clean$bystuwt.x, na.rm = TRUE)))),
                               case_wts_rounded)
  ) %>%
  select(-is_synthetic, -.original_row, -.smote_row) %>%
  as_tibble()

# Verify final balance
cat("\n=== FINAL SMOTE DATASET ===\n")
els_train_up |> count(dropout_factor) |> mutate(prop = n / sum(n))

cat("\n=== SAMPLE SIZE COMPARISON ===\n")
cat("Original training (clean):", nrow(els_train_clean), "\n")
cat("SMOTE training:", nrow(els_train_up), "\n")
cat("Synthetic rows added:", nrow(els_train_up) - nrow(els_train_clean), "\n")

#____________________________
# Verify factor levels are correct
cat("\n=== Verify factor levels ===\n")
cat("bysex levels:", levels(els_train_up$bysex), "\n")
cat("byrace levels:", levels(els_train_up$byrace), "\n")
cat("Sample Likert variable (BYS20A):", levels(els_train_up$BYS20A), "\n")
cat("Sample frequency variable (BYS22A):", levels(els_train_up$BYS22A), "\n")

#____________________________
# Skim the SMOTE data to verify distributions
els_train_up |> 
  select(-case_wts, -case_wts_scaled, -case_wts_rounded, 
         -STU_ID, -SCH_ID, -STRAT_ID, -psu) |>
  skim()

#____________________________
# Save SMOTE training data
saveRDS(els_train_up, "els_train_up_smote.rds")

# To compare with original upsampling later:
# els_train_up_original <- readRDS("els_train_up.rds")
# els_train_up_smote <- readRDS("els_train_up_smote.rds")

#_______________________________________________________________________________
#_______________________________________________________________________________

#_______________________________________________________________________________

#_______________________________________________________________________________










#_______________________________________________________________________________
#_______________________________________________________________________________
#####DOWNSAMPLE####

# Following the same approach as upsampling but reducing majority class
# to match minority class size (preventing data leakage by only applying to training)

# --- Clean training: drop rows with ANY missing values ---
els_train_clean <- els_train %>%
  drop_na()

# --- Check base counts ---
els_train_clean |> count(dropout_factor) |> mutate(prop = n / sum(n))

# DOWNSAMPLE training to 1:1 (keeps all minority rows; samples majority without replacement)
# (No leakage: only applied to training data)

# Counts by class
ctr <- els_train_clean |> count(dropout_factor) 

# Find minority class (smallest count)
min_class <- ctr |> 
  slice_min(n, n = 1, with_ties = FALSE) |> 
  pull(dropout_factor) |>
  as.character()

# Get minority class count
n_min <- ctr |> 
  slice_min(n, n = 1, with_ties = FALSE) |> 
  pull(n)

# Split train by class (dropping NA level if present)
spl <- split(els_train_clean, els_train_clean$dropout_factor, drop = TRUE)

# Keep all minority class observations
min_df <- spl[[min_class]]

# Sample majority class down to minority size
maj_df <- bind_rows(spl[setdiff(names(spl), min_class)]) |>
  slice_sample(n = n_min, replace = FALSE)  # without replacement for downsampling

# Bind balanced train (downsampled)
els_train_down <- bind_rows(min_df, maj_df) |> as_tibble()

# Verify balance
print("Downsampled training set balance:")
els_train_down |> count(dropout_factor) |> mutate(prop = n / sum(n))

# Compare to original and upsampled
print("Original training set balance:")
els_train_clean |> count(dropout_factor) |> mutate(prop = n / sum(n))

print("Training set sizes:")
cat("Original training (clean):", nrow(els_train_clean), "\n")
cat("Upsampled training:", nrow(els_train_up), "\n") 
cat("Downsampled training:", nrow(els_train_down), "\n")



# Save downsampled training data  
saveRDS(els_train_down, "els_train_down.rds")

# Load them back later
els_train_down <- readRDS("els_train_down.rds")
#_______________________________________________________________________________








#_______________________________________________________________________________
##### CV folds *on the downsampled training set*####
els_folds_down <- vfold_cv(els_train_down, v = 5, strata = dropout_factor)
#_______________________________________________________________________________






#_______________________________________________________________________________
##### CV folds *on the upsampled training set*####
els_folds_up <- vfold_cv(els_train_up, v = 5, strata = dropout_factor)
#_______________________________________________________________________________



#_______________________________________________________________________________
#Create test metrics to assess with:
# Create test metrics object
test_metrics <- metric_set(
  yardstick::accuracy, 
  yardstick::roc_auc, 
  yardstick::sens, 
  yardstick::spec, 
  yardstick::f_meas
)





#_______________________________________________________________________________
#####RECIPES####
#Do not use recipes! The weights are not compatible with the workflows#
# Case weights are handled separately in the workflow
# Proceed with your recipe/workflow, using els_train_up as the recipe data
dropout_recipe <- recipe(
  dropout_factor ~ BYS20A + BYS20B + BYS20D + BYS20E + 
    BYS20F + BYS20G + BYS20J + BYS20K + 
    BYS21B + BYS21C + BYS27E + BYS27F + 
    BYS27H + bysex + byrace,
  data = els_train_up
) |>
  step_naomit(all_predictors(), all_outcomes()) |>
  step_mutate(across(where(is.character), factor)) |>
  step_dummy(all_nominal_predictors())

#____________________________
#Examine:

# Prep and bake the recipe
rec_prep <- dropout_recipe |> prep()

balanced_data <- rec_prep |> bake(new_data = NULL)

# Check the new class distribution
balanced_data |>
  count(dropout_factor) |>
  mutate(prop = n / sum(n))

#_______________________________________________________________________________






#_______________________________________________________________________________
##### BASELINE MODEL - ALWAYS PREDICT "NOT DROPOUT" ####
# It is using the test data; no training; predict as is.
#_______________________________________________________________________________

# Create baseline predictions (always predict majority class)
baseline_preds <- els_test_clean %>%
  select(dropout_factor, case_wts_rounded) %>%
  mutate(.pred_class = factor("Not_Dropout", levels = levels(dropout_factor)))

# Calculate baseline metrics
baseline_metrics <- yardstick::metric_set(
  yardstick::accuracy, yardstick::sens, yardstick::spec,
  yardstick::precision, yardstick::recall, yardstick::f_meas
)(
  baseline_preds,
  truth = dropout_factor,
  estimate = .pred_class,
  case_weights = case_wts_rounded
)

print("=== Baseline Model Performance (Always Predict Not_Dropout) ===")
print(baseline_metrics)

# Baseline confusion matrix
baseline_conf_matrix <- yardstick::conf_mat(
  baseline_preds,
  truth = dropout_factor,
  estimate = .pred_class,
  case_weights = case_wts_rounded
)

print("=== Baseline Confusion Matrix ===")
print(baseline_conf_matrix)

# Show what percentage of students actually don't drop out
print("=== Class Distribution in Test Set ===")
class_dist <- baseline_preds %>%
  count(dropout_factor, wt = as.numeric(case_wts_rounded)) %>%
  mutate(percentage = round(100 * n / sum(n), 1))
print(class_dist)








#_______________________________________________________________________________

#_______________________________________________________________________________

#_______________________________________________________________________________

#_______________________________________________________________________________



#### Logistic Regression ####


#####Logistic Regression Train (upsampled) ####

#____________________________
# LR Model specifications
logistic_spec <- logistic_reg() %>% 
  set_engine("glm") %>%
  set_mode("classification")


#____________________________
#LR Workflow
# Following documentation: add_case_weights -> add_formula -> add_model
logistic_wf <- workflow() %>%
  add_case_weights(case_wts_rounded) %>%
  add_formula(dropout_factor ~ BYS20A + BYS20B + BYS20C + BYS20D + BYS20E + 
                BYS20F + BYS20G + BYS20H + BYS20I + BYS20J + BYS20K + 
                BYS20L + BYS20M + BYS20N + BYS21A + BYS21B + BYS21C + 
                BYS21D + BYS21E + BYS22A + BYS22B + BYS22C + BYS22D + 
                BYS22E + BYS22G + BYS22H + BYS27H + bysex + byrace) %>%
  add_model(logistic_spec)



#____________________________
#FIT LR
# Fit logistic regression directly (case weights handled by workflow)
logistic_cv_up_results <- fit_resamples(
  logistic_wf,
  resamples = els_folds_up,
  metrics = metric_set(accuracy, roc_auc, yardstick::sens, yardstick::spec, 
                       yardstick::f_meas),
  control = control_resamples(save_pred = TRUE)
)

#_______________________________________________________________________________





#_______________________________________________________________________________
####SAVE/LOAD The Log Reg initially trained upsample model ####

# Save the CV results to file
saveRDS(logistic_cv_up_results, file = "logistic_cv_up_results.rds")

# Later (in a fresh session), load it back
logistic_cv_up_results <- readRDS("logistic_cv_up_results.rds")


#_______________________________________________________________________________








#_______________________________________________________________________________

##### LOGIISTIC REGRESSION Upsample Examination CVs####
#_______________________________________________________________________________
#EVALUATE TRAINING OOF LOGISTIC REGRESSION
#_______________________________________________________________________________

#COLLECT METRICS
collect_metrics(logistic_cv_up_results)
#_______________________________________________________________________________





#_______________________________________________________________________________
# See performance for each individual fold
collect_metrics(logistic_cv_up_results, summarize = FALSE)
#_______________________________________________________________________________






#_______________________________________________________________________________
#ROC AUC For LOGISTIC REGRESSION CVs

# Collect fold-level predictions
cv_preds_up <- collect_predictions(logistic_cv_up_results)


# Check factor levels
levels(cv_preds_up$dropout_factor)
# Shows  "Dropout" "Not_Dropout"

# ROC curve + AUC with correct event
roc_tbl_up <- roc_curve(cv_preds_up,
                     truth = dropout_factor,
                     .pred_Dropout)

autoplot(roc_tbl_up) + ggtitle("ROC (Dropout as positive)")

roc_auc(cv_preds_up,
        truth = dropout_factor,
        .pred_Dropout)
#_______________________________________________________________________________






#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________
##### PREDICT ON TEST SET - LOGISTIC REGRESSION####
#_______________________________________________________________________________


#____________________________
#FIT on TEST

#Final LR Fit on the full data
# Fit the logistic regression on the full balanced training data
final_logistic_up_fit <- fit(logistic_wf, data = els_train_up)


#____________________________
#SAVE/LOAD

# Save the fitted model so you don’t need to retrain each run
saveRDS(final_logistic_up_fit, "final_logistic_up_fit.rds")

# To reload in a future session:
final_logistic_up_fit <- readRDS("final_logistic_up_fit.rds")



#____________________________
#PREDICT TESTSET

# Make predictions on test set
test_predictions_up <- predict(final_logistic_up_fit, els_test_clean, type = "prob") |>
  dplyr::bind_cols(
    predict(final_logistic_up_fit, els_test_clean, type = "class")
  ) |>
  dplyr::bind_cols(
    els_test_clean |> dplyr::select(dropout_factor, case_wts_rounded)
  )

# View structure of predictions
glimpse(test_predictions_up)
#_______________________________________________________________________________







#_______________________________________________________________________________
##### EVALUATE TEST SET PERFORMANCE LOGIISTIC REGRESSION####
#_______________________________________________________________________________


# Calculate the SAME metrics as cross-validation: accuracy, roc_auc, sens, spec
test_metrics <- metric_set(accuracy, roc_auc, yardstick::sens, yardstick::spec,
                           yardstick::f_meas)

test_results_up <- test_predictions_up |>
  test_metrics(truth = dropout_factor,
               estimate = .pred_class,
               .pred_Dropout,
               case_weights = case_wts_rounded)


print("Test Set Performance (Same metrics as CV):")
test_results_up
#_______________________________________________________________________________

#_______________________________________________________________________________





#_______________________________________________________________________________


#_______________________________________________________________________________
##### THRESHOLD SWEEP - LOGISTIC REGRESSION UPSAMPLED ####
#_______________________________________________________________________________



# Test different probability thresholds
threshold_grid <- tibble::tibble(threshold = seq(0.05, 0.5, by = 0.01))

logistic_threshold_eval_up <- threshold_grid %>%
  mutate(
    metrics = map(
      threshold,
      ~ {
        logistic_preds <- test_predictions_up %>%
          mutate(
            .pred_class = factor(
              ifelse(.pred_Dropout >= .x, "Dropout", "Not_Dropout"),
              levels = levels(dropout_factor)
            )
          )
        test_metrics(
          logistic_preds,
          truth        = dropout_factor,
          estimate     = .pred_class,
          .pred_Dropout,
          case_weights = case_wts_rounded
        )
      }
    )
  ) %>%
  unnest(metrics)

# View results with available metrics only
logistic_threshold_eval_up %>%
  filter(.metric %in% c("sens", "spec", "f_meas")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(across(c(sens, spec, f_meas), ~ round(.x, 3))) %>%
  print(n = Inf)






#_______________________________________________________________________________

#_______________________________________________________________________________
 #END LOGISTIC REGRESSION SECTION UPSAMPLE
#_______________________________________________________________________________













#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________




#####Logistic Regression (downsampled) ####
#____________________________
# LR Model specifications (same as upsampled)
logistic_spec <- logistic_reg() %>% 
  set_engine("glm") %>%
  set_mode("classification")





#____________________________
#LR Workflow
# Following documentation: add_case_weights -> add_formula -> add_model
logistic_wf_down <- workflow() %>%
  add_case_weights(case_wts_rounded) %>%
  add_formula(dropout_factor ~ BYS20A + BYS20B + BYS20C + BYS20D + BYS20E + 
                BYS20F + BYS20G + BYS20H + BYS20I + BYS20J + BYS20K + 
                BYS20L + BYS20M + BYS20N + BYS21A + BYS21B + BYS21C + 
                BYS21D + BYS21E + BYS22A + BYS22B + BYS22C + BYS22D + 
                BYS22E + BYS22G + BYS22H + BYS27H + bysex + byrace) %>%
  add_model(logistic_spec)





#____________________________
#FIT LR
# Fit logistic regression directly (case weights handled by workflow)
logistic_cv_down_results <- fit_resamples(
  logistic_wf_down,
  resamples = els_folds_down,
  metrics = test_metrics,
  control = control_resamples(save_pred = TRUE)
)
#_______________________________________________________________________________
#_______________________________________________________________________________








#####SAVE/LOAD The LOG REG downsample model#####

# Save the CV results to file
saveRDS(logistic_cv_down_results, file = "logistic_cv_down_results.rds")
# Later (in a fresh session), load it back
logistic_cv_down_results <- readRDS("logistic_cv_down_results.rds")
#_______________________________________________________________________________
#_______________________________________________________________________________






##### LOGIISTIC REGRESSION Downsample Examination CVs####
#_______________________________________________________________________________
#EVALUATE TRAINING OOF LOGISTIC REGRESSION
#_______________________________________________________________________________
#COLLECT METRICS
collect_metrics(logistic_cv_down_results)
#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________



#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________
##### PREDICT ON TEST SET - LOGISTIC REGRESSION DOWNSAMPLED####
#_______________________________________________________________________________


#____________________________
#FIT on TEST

#Final LR Fit on the full data
# Fit the logistic regression on the full balanced training data
final_logistic_down_fit <- fit(logistic_wf_down, data = els_train_down)


#____________________________
####SAVE/LOAD The Final LOG Reg Model Fully Trained (Downsampled)####

# Save the fitted model so you don't need to retrain each run
saveRDS(final_logistic_down_fit, "final_logistic_down_fit.rds")

# To reload in a future session:
final_logistic_down_fit <- readRDS("final_logistic_down_fit.rds")



#____________________________
#PREDICT TESTSET

# Make predictions on test set
test_predictions_down <- predict(final_logistic_down_fit, els_test_clean, type = "prob") |>
  dplyr::bind_cols(
    predict(final_logistic_down_fit, els_test_clean, type = "class")
  ) |>
  dplyr::bind_cols(
    els_test_clean |> dplyr::select(dropout_factor, case_wts_rounded)
  )

# View structure of predictions
glimpse(test_predictions_down)
#_______________________________________________________________________________







#_______________________________________________________________________________
##### EVALUATE TEST SET PERFORMANCE LOGISTIC REGRESSION DOWNSAMPLED####
#_______________________________________________________________________________

# test_metrics to include F1 for proper comparison
test_metrics <- metric_set(accuracy, roc_auc, yardstick::sens, yardstick::spec, yardstick::f_meas)

# Evaluate downsampled test performance
test_results_down <- test_predictions_down |>
  test_metrics(truth = dropout_factor,
               estimate = .pred_class,
               .pred_Dropout,
               case_weights = case_wts_rounded)

print("Test Set Performance (Same metrics as CV) - Downsampled:")
test_results_down

#_______________________________________________________________________________

#_______________________________________________________________________________

#_______________________________________________________________________________

# Test different probability thresholds for downsampled model
threshold_grid <- tibble::tibble(threshold = seq(0.05, 0.5, by = 0.01))

logistic_threshold_eval_down <- threshold_grid %>%
  mutate(
    metrics = map(
      threshold,
      ~ {
        logistic_preds <- test_predictions_down %>%
          mutate(
            .pred_class = factor(
              ifelse(.pred_Dropout >= .x, "Dropout", "Not_Dropout"),
              levels = levels(dropout_factor)
            )
          )
        test_metrics(
          logistic_preds,
          truth        = dropout_factor,
          estimate     = .pred_class,
          .pred_Dropout,
          case_weights = case_wts_rounded
        )
      }
    )
  ) %>%
  unnest(metrics)

# View how sensitivity, specificity, F1, etc. change across thresholds
logistic_threshold_eval_down %>%
  filter(.metric %in% c("sens", "spec", "f_meas")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(across(c(sens, spec, f_meas), ~ round(.x, 3))) %>%
  print(n = Inf)











#_______________________________________________________________________________
# #####Penalized Regression (Upsampled, untuned) ####
#_______________________________________________________________________________

#UNTUNED#
#____________________________
#____________________________
# Penalized Model specifications (default parameters)
penalized_spec_default <- logistic_reg(
  penalty = 0.01,    # Small default penalty
  mixture = 0        # 1= LASSO, .5 = Elastic, 0 = Ridge
) %>% 
  set_engine("glmnet") %>%
  set_mode("classification")

#____________________________
# Penalized Workflow (untuned)
penalized_wf_default <- workflow() %>%
  add_case_weights(case_wts_rounded) %>%
  add_formula(dropout_factor ~ BYS20A + BYS20B + BYS20C + BYS20D + BYS20E + 
                BYS20F + BYS20G + BYS20H + BYS20I + BYS20J + BYS20K + 
                BYS20L + BYS20M + BYS20N + BYS21A + BYS21B + BYS21C + 
                BYS21D + BYS21E + BYS22A + BYS22B + BYS22C + BYS22D + 
                BYS22E + BYS22G + BYS22H + BYS27H + bysex + byrace) %>%
  add_model(penalized_spec_default)

#____________________________
# Fit with cross-validation (untuned)
penalized_cv_default_results <- fit_resamples(
  penalized_wf_default,
  resamples = els_folds_up,
  metrics = test_metrics,
  control = control_resamples(save_pred = TRUE)
)

# View CV results
collect_metrics(penalized_cv_default_results)

# Fit final model
penalized_final_default <- fit(penalized_wf_default, data = els_train_up)

# Save untuned model
saveRDS(penalized_cv_default_results, "penalized_cv_default_results.rds")
saveRDS(penalized_final_default, "penalized_final_default.rds")



#____________________________
# Test set predictions
penalized_test_predictions <- predict(penalized_final_default, els_test_clean, type = "prob") |>
  bind_cols(predict(penalized_final_default, els_test_clean, type = "class")) |>
  bind_cols(els_test_clean |> select(dropout_factor, case_wts_rounded))

# Test set evaluation
penalized_test_results <- penalized_test_predictions |>
  test_metrics(truth = dropout_factor,
               estimate = .pred_class,
               .pred_Dropout,
               case_weights = case_wts_rounded)

print("Ridge Regression Test Set Performance:")
print(penalized_test_results)


# Save fully trained model (this was missing)
saveRDS(penalized_final_default, "penalized_final_default.rds")


# Threshold sweep for Ridge
threshold_grid <- tibble::tibble(threshold = seq(0.05, 0.5, by = 0.01))

penalized_threshold_eval <- threshold_grid %>%
  mutate(
    metrics = map(
      threshold,
      ~ {
        penalized_preds <- penalized_test_predictions %>%
          mutate(
            .pred_class = factor(
              ifelse(.pred_Dropout >= .x, "Dropout", "Not_Dropout"),
              levels = levels(dropout_factor)
            )
          )
        test_metrics(
          penalized_preds,
          truth        = dropout_factor,
          estimate     = .pred_class,
          .pred_Dropout,
          case_weights = case_wts_rounded
        )
      }
    )
  ) %>%
  unnest(metrics)

# View threshold results
penalized_threshold_eval %>%
  filter(.metric %in% c("sens", "spec", "f_meas")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(across(c(sens, spec, f_meas), ~ round(.x, 3))) %>%
  print(n = Inf)

# Save test predictions
saveRDS(penalized_test_predictions, "penalized_test_predictions.rds")










#_______________________________________________________________________________

#PEnalized Regression Upsampled Tuned
#____________________________
# Penalized Model specifications
penalized_spec_up <- logistic_reg(penalty = tune(), mixture = 0) %>% 
  set_engine("glmnet") %>%
  set_mode("classification")

#____________________________
#Penalized Workflow
penalized_wf_up <- workflow() %>%
  add_case_weights(case_wts_rounded) %>%
  add_formula(dropout_factor ~ BYS20A + BYS20B + BYS20C + BYS20D + BYS20E + 
                BYS20F + BYS20G + BYS20H + BYS20I + BYS20J + BYS20K + 
                BYS20L + BYS20M + BYS20N + BYS21A + BYS21B + BYS21C + 
                BYS21D + BYS21E + BYS22A + BYS22B + BYS22C + BYS22D + 
                BYS22E+ BYS22G + BYS22H + BYS27H + bysex + byrace) %>%
  add_model(penalized_spec_up)


# Log-spaced penalty grid (wider search; adjust levels as needed)
pen_grid <- dials::grid_regular(
  # PENALTY PARAMETER SPECIFICATION
  # The penalty parameter controls the strength of regularization.
  dials::penalty(
    
    # Log10 range from 0.0001 to 0.1
    range = c(-4, -1),                    
    
    
    # Log10 transformation - space the values out logarithmically not linearly
    trans = scales::log10_trans()                       # No pre-specified values (use range instead)
  ),
  
  # GRID GENERATION DEFAULTS (all hidden in your code):
  
  # The algorithm will create 30 evenly-spaced points between your range endpoints
  # Each point represents a different penalty value to test
  
  levels = 30                            # 30 penalty values to test
  # filter = NULL,                        # HIDDEN: No filtering of parameter combinations
  # complete = TRUE,                      # HIDDEN: Include all combinations (not applicable for 1 parameter)
  
  # SIZE CONTROL DEFAULTS:
  # size = NULL                           # HIDDEN: Not used when levels is specified
)

# WHAT THIS CREATES:
# - 30 penalty values logarithmically spaced from 10^-4 to 10^-1
# - Actual penalty values: 0.0001, 0.0001259, 0.0001585, ..., 0.0631, 0.0794, 0.1
# - Each value will be tested in cross-validation
# - Grid search will fit 30 × 5 folds = 150 total model fits
# - Best penalty will be selected based on your specified metric (roc_auc)

#____________________________
#____________________________
#TUNE Penalized
penalized_tune_up <- tune_grid(
  penalized_wf_up,
  resamples = els_folds_up,
  grid = pen_grid,
  metrics = metric_set(accuracy, roc_auc, yardstick::sens, yardstick::spec, yardstick::f_meas),
  control = control_grid(save_pred = TRUE)
)


collect_metrics(penalized_tune_up)

# See best performing configurations
show_best(penalized_tune_up, metric = "roc_auc", n = 5)
show_best(penalized_tune_up, metric = "f_meas", n = 5)


# Get the best penalty value
best_penalty <- select_best(penalized_tune_up, metric = "roc_auc")
best_penalty

# See all metrics for the best model
collect_metrics(penalized_tune_up) %>%
  filter(penalty == best_penalty$penalty) %>%
  select(.metric, mean, std_err) %>%
  mutate(mean = round(mean, 3), std_err = round(std_err, 4))

#____________________________
# ____________________________
# SAVE tuned LASSO results
saveRDS(penalized_tune_up, file = "penalized_tune_up.rds")
saveRDS(penalized_wf_up, file = "penalized_wf_up.rds")

# ____________________________
# LOAD tuned LASSO results
penalized_tune_up <- readRDS("penalized_tune_up.rds")

#_______________________________________________________________________________










#_______________________________________________________________________________
# Penalized REGRESSION UP SAmple Tuned FULLY TRAIN AND PREDICT ON TEST DATA

#_______________________________________________________________________________
##### PENALIZED REGRESSION FULLY TRAIN AND PREDICT ON TEST DATA - UPSAMPLED ####
#_______________________________________________________________________________

# Select best hyperparameters and finalize workflow
penalized_best_params <- select_best(penalized_tune_up, metric = "roc_auc")
penalized_final_workflow <- finalize_workflow(penalized_wf_up, penalized_best_params)

# Fit final model on full training data
penalized_final_fit <- fit(penalized_final_workflow, data = els_train_up)

# Save final model
saveRDS(penalized_final_fit, "penalized_final_fit_tuned.rds")

# Test set predictions
penalized_test_predictions_tuned <- predict(penalized_final_fit, els_test_clean, type = "prob") |>
  bind_cols(predict(penalized_final_fit, els_test_clean, type = "class")) |>
  bind_cols(els_test_clean |> select(dropout_factor, case_wts_rounded))

# Test set evaluation
penalized_test_results_tuned <- penalized_test_predictions_tuned |>
  metric_set(accuracy, roc_auc, yardstick::sens, yardstick::spec, yardstick::f_meas)(
    truth = dropout_factor,
    estimate = .pred_class,
    .pred_Dropout,
    case_weights = case_wts_rounded
  )

print("Tuned Ridge Regression Test Set Performance:")
print(penalized_test_results_tuned)

# Save test predictions
saveRDS(penalized_test_predictions_tuned, "penalized_test_predictions_tuned.rds")
# ____________________________

#_______________________________________________________________________________

#_______________________________________________________________________________
##### THRESHOLD SWEEP - PENALIZED REGRESSION Tuned UPSAMPLED ####
## Fully Trained Model
#_______________________________________________________________________________

# Threshold sweep for tuned Ridge regression
threshold_grid <- tibble::tibble(threshold = seq(0.05, 0.5, by = 0.01))

penalized_threshold_eval_tuned <- threshold_grid %>%
  mutate(
    metrics = map(
      threshold,
      ~ {
        penalized_preds <- penalized_test_predictions_tuned %>%
          mutate(
            .pred_class = factor(
              ifelse(.pred_Dropout >= .x, "Dropout", "Not_Dropout"),
              levels = levels(dropout_factor)
            )
          )
        metric_set(accuracy, roc_auc, yardstick::sens, yardstick::spec, yardstick::f_meas)(
          penalized_preds,
          truth        = dropout_factor,
          estimate     = .pred_class,
          .pred_Dropout,
          case_weights = case_wts_rounded
        )
      }
    )
  ) %>%
  unnest(metrics)

# View threshold results
penalized_threshold_eval_tuned %>%
  filter(.metric %in% c("sens", "spec", "f_meas")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(across(c(sens, spec, f_meas), ~ round(.x, 3))) %>%
  print(n = Inf)

# Save threshold results
saveRDS(penalized_threshold_eval_tuned, "penalized_threshold_eval_tuned.rds")


#_______________________________________________________________________________
# End PENALIZED REGRESSION Tuned UPSAMPLED 
#_______________________________________________________________________________

#_______________________________________________________________________________
#_______________________________________________________________________________





#_______________________________________________________________________________

#_______________________________________________________________________________

#_______________________________________________________________________________

#_______________________________________________________________________________
##### Random Forest (Upsampled, untuned) ####
#_______________________________________________________________________________

#UNTUNED - DEFAULT SETTINGS#
#____________________________

# Set up parallel processing with 7 cores
library(doParallel)
library(parallel)

cl <- makeCluster(7)
registerDoParallel(cl)

#____________________________
# Random Forest Model specifications (default parameters)
rf_spec_default_up <- rand_forest(trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

#____________________________
# Random Forest Workflow (untuned)
rf_wf_default_up <- workflow() %>%
  add_case_weights(case_wts_rounded) %>%
  add_formula(
    dropout_factor ~ BYS20A + BYS20B + BYS20C + BYS20D + BYS20E + 
      BYS20F + BYS20G + BYS20H + BYS20I + BYS20J + BYS20K + 
      BYS20L + BYS20M + BYS20N + BYS21A + BYS21B + BYS21C + 
      BYS21D + BYS21E + BYS22A + BYS22B + BYS22C + BYS22D + 
      BYS22E + BYS22G + BYS22H + BYS27H + bysex + byrace
  ) %>%
  add_model(rf_spec_default_up)

#____________________________
# Fit with cross-validation (untuned)
rf_cv_default_results_up <- fit_resamples(
  rf_wf_default_up,
  resamples = els_folds_up,
  metrics = test_metrics,
  control = control_resamples(save_pred = TRUE)
)

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

# View CV results
collect_metrics(rf_cv_default_results_up)

# Fit final model
rf_final_default_up <- fit(rf_wf_default_up, data = els_train_up)

# Save untuned model
saveRDS(rf_cv_default_results_up, "rf_cv_default_results_up.rds")
saveRDS(rf_final_default_up, "rf_final_default_up.rds")

#____________________________
# Test set predictions
rf_test_predictions_up <- predict(rf_final_default_up, els_test_clean, type = "prob") |>
  bind_cols(predict(rf_final_default_up, els_test_clean, type = "class")) |>
  bind_cols(els_test_clean |> select(dropout_factor, case_wts_rounded))

# Test set evaluation
rf_test_results_up <- rf_test_predictions_up |>
  test_metrics(truth = dropout_factor,
               estimate = .pred_class,
               .pred_Dropout,
               case_weights = case_wts_rounded)

print("Random Forest Test Set Performance (Upsampled, Untuned):")
print(rf_test_results_up)

#____________________________
# Threshold sweep for Random Forest
threshold_grid <- tibble::tibble(threshold = seq(0.05, 0.5, by = 0.01))

rf_threshold_eval_up <- threshold_grid %>%
  mutate(
    metrics = map(
      threshold,
      ~ {
        rf_preds <- rf_test_predictions_up %>%
          mutate(
            .pred_class = factor(
              ifelse(.pred_Dropout >= .x, "Dropout", "Not_Dropout"),
              levels = levels(dropout_factor)
            )
          )
        test_metrics(
          rf_preds,
          truth        = dropout_factor,
          estimate     = .pred_class,
          .pred_Dropout,
          case_weights = case_wts_rounded
        )
      }
    )
  ) %>%
  unnest(metrics)

# View threshold results
rf_threshold_eval_up %>%
  filter(.metric %in% c("sens", "spec", "f_meas")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(across(c(sens, spec, f_meas), ~ round(.x, 3))) %>%
  print(n = Inf)

# Save test predictions
saveRDS(rf_test_predictions_up, "rf_test_predictions_up.rds")

#_______________________________________________________________________________





































#_______________________________________________________________________________

#_______________________________________________________________________________

#_______________________________________________________________________________

#_______________________________________________________________________________
#Random Forest (Upsampled,tuned, training data )
#_______________________________________________________________________________



library(doParallel)
library(parallel)

# Set up parallel processing with 7 cores
cl <- makeCluster(7)
registerDoParallel(cl)



# Tunable Random Forest spec
rf_spec_tuned <- rand_forest(
  trees = 1000,          # Reduce from 2000 (faster, similar performance)
  mtry = tune(),         
  min_n = tune()         
) %>%
  set_engine("ranger", 
             max.depth = tune(),
             sample.fraction = tune()) %>%  # ← ADD THIS: tune subsampling
  set_mode("classification")

# Workflow
rf_workflow_tuned_up <- workflow() %>%
  add_case_weights(case_wts_rounded) %>%
  add_formula(
    dropout_factor ~ BYS20A + BYS20B + BYS20C + BYS20D + BYS20E + 
      BYS20F + BYS20G + BYS20H + BYS20I + BYS20J + BYS20K + 
      BYS20L + BYS20M + BYS20N + BYS21A + BYS21B + BYS21C + 
      BYS21D + BYS21E + BYS22A + BYS22B + BYS22C + BYS22D + 
      BYS22E + BYS22G + BYS22H + BYS27H + bysex + byrace
  ) %>%
  add_model(rf_spec_tuned)

#____________________________
#create a grid for the search
rf_grid <- expand_grid(
  
  # mtry: Number of predictors randomly sampled at each split from all 29.
  # Controls model complexity - lower = more randomness/less correlation between trees
  # Adjust: Use ~sqrt(p) for classification (here √29 ≈ 5-6) as rule of thumb, test smaller for more diversity 
  mtry = c(2, 3, 4, 5),           # 3 values for mtry
  
  # min_n: Minimum observations required in a node to attempt a split
  # Controls overfitting - higher = simpler trees, less likely to memorize noise
  # Adjust: Increase if overfitting (trees too complex), decrease if underfitting
  # 5 Allows small nodes → More complex trees; 20 Forces larger nodes → Simpler trees
  min_n = c(10, 20, 40),         
  
  # max.depth: Maximum number of levels in each tree
  # Controls tree complexity - lower = shallower trees, less overfitting
  # Adjust: Keep low (1-3) with limited samples to prevent memorizing training data
  max.depth = c(3 ,4, 5),
  
  sample.fraction = c(0.6, 0.8)

)

# Tune the model
rf_tune_results <- tune_grid(
  rf_workflow_tuned_up,
  resamples = els_folds_up,
  grid = rf_grid,
  metrics = test_metrics,
  control = control_grid(save_pred = TRUE)
)

# See best results
show_best(rf_tune_results, metric = "sens", n = 5)  # Best for sensitivity
show_best(rf_tune_results, metric = "roc_auc", n = 5)  # Best overall

# See all metrics for all parameter combinations
collect_metrics(rf_tune_results) %>%
  select(mtry, min_n, max.depth, sample.fraction, .metric, mean) %>%  # Added sample.fraction
  pivot_wider(names_from = .metric, values_from = mean) %>%
  arrange(desc(sens)) %>%
  mutate(across(c(accuracy, roc_auc, sens, spec, f_meas), ~ round(.x, 3))) %>%
  print(n = Inf)


# Stop parallel processing and return to sequential
stopCluster(cl)
registerDoSEQ()

#save
saveRDS(rf_tune_results, "rf_tune_results.rds")  # Contains all tuning info
# Also save the workflow for reference
saveRDS(rf_workflow_tuned_up, "rf_workflow_tuned_up.rds")

#load
rf_tune_results <- readRDS("rf_tune_results.rds")








#_______________________________________________________________________________
##### RANDOM FOREST - OOF EXAMINATION ####
#_______________________________________________________________________________

# Get ALL metrics for the best ROC AUC model
select_best(rf_tune_results, metric = "roc_auc") %>%
  inner_join(collect_metrics(rf_tune_results), by = ".config") %>%
  select(.metric, mean, std_err) %>%
  mutate(across(c(mean, std_err), ~ round(.x, 3)))























#_______________________________________________________________________________
##### RANDOM FOREST - FIT FINAL MODEL AND TEST SET EVALUATION ####
#_______________________________________________________________________________

#____________________________
# 1. SELECT BEST HYPERPARAMETERS
rf_best_params_up <- select_best(rf_tune_results, metric = "roc_auc")
print("Best Random Forest hyperparameters:")
print(rf_best_params_up)

#____________________________
# 2. FINALIZE WORKFLOW WITH BEST PARAMETERS
rf_final_workflow_up <- finalize_workflow(rf_workflow_tuned_up, rf_best_params_up)

#____________________________
# 3. FIT FINAL MODEL ON FULL TRAINING DATA
rf_final_fit_up <- fit(rf_final_workflow_up, data = els_train_up)

# Save final model
saveRDS(rf_final_fit_up, "rf_final_fit_up.rds")
saveRDS(rf_best_params_up, "rf_best_params_up.rds")

# To reload later:
rf_final_fit_up <- readRDS("rf_final_fit_up.rds")

#____________________________
# 4. MAKE PREDICTIONS ON TEST SET
rf_test_predictions_up <- predict(rf_final_fit_up, els_test_clean, type = "prob") |>
  bind_cols(predict(rf_final_fit_up, els_test_clean, type = "class")) |>
  bind_cols(els_test_clean |> select(dropout_factor, case_wts_rounded))

#____________________________
# 5. EVALUATE TEST SET PERFORMANCE
rf_test_results_up <- rf_test_predictions_up |>
  test_metrics(truth = dropout_factor,
               estimate = .pred_class,
               .pred_Dropout,
               case_weights = case_wts_rounded)

print("Random Forest Test Set Performance (Tuned, Upsampled):")
print(rf_test_results_up)

# Save test predictions
saveRDS(rf_test_predictions_up, "rf_test_predictions_up.rds")

#____________________________
# 6. THRESHOLD SWEEP
threshold_grid <- tibble::tibble(threshold = seq(0.05, 0.5, by = 0.01))

rf_threshold_eval_up <- threshold_grid %>%
  mutate(
    metrics = map(
      threshold,
      ~ {
        rf_preds <- rf_test_predictions_up %>%
          mutate(
            .pred_class = factor(
              ifelse(.pred_Dropout >= .x, "Dropout", "Not_Dropout"),
              levels = levels(dropout_factor)
            )
          )
        test_metrics(
          rf_preds,
          truth        = dropout_factor,
          estimate     = .pred_class,
          .pred_Dropout,
          case_weights = case_wts_rounded
        )
      }
    )
  ) %>%
  unnest(metrics)

#____________________________
# 7. VIEW THRESHOLD SWEEP RESULTS
rf_threshold_eval_up %>%
  filter(.metric %in% c("sens", "spec", "f_meas")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(across(c(sens, spec, f_meas), ~ round(.x, 3))) %>%
  print(n = Inf)

# Save threshold results
saveRDS(rf_threshold_eval_up, "rf_threshold_eval_up.rds")

#_______________________________________________________________________________

#end RF TUNED


#_______________________________________________________________________________






#_______________________________________________________________________________

#_______________________________________________________________________________
##### SVM LINEAR (Upsampled, Untuned) ####
#_______________________________________________________________________________

# Load required library
library(kernlab)  # For SVM engine
library(doParallel)
library(parallel)

# Set up parallel processing
cl <- makeCluster(7)
registerDoParallel(cl)

#____________________________
# SVM Model specification (using defaults) LINEAR
svm_spec_default_up <- svm_poly(
  cost = 1,           # Regularization parameter (default)
  degree = 1,         # Polynomial degree (1 = linear SVM)
  scale_factor = 1,   # Scaling factor for polynomial kernel (default)
  margin = NULL       # Margin parameter (NULL = use algorithm default)
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

#____________________________
# SVM Linear Workflow (untuned)
svm_linear_wf_default_up <- workflow() %>%
  add_formula(
    dropout_factor ~ BYS20A + BYS20B + BYS20C + BYS20D + BYS20E + 
      BYS20F + BYS20G + BYS20H + BYS20I + BYS20J + BYS20K + 
      BYS20L + BYS20M + BYS20N + BYS21A + BYS21B + BYS21C + 
      BYS21D + BYS21E + BYS22A + BYS22B + BYS22C + BYS22D + 
      BYS22E + BYS22G + BYS22H + BYS27H + bysex + byrace
    # Note: BYS22F removed (near-zero variance)
  ) %>%
  add_model(svm_spec_default_up)

#____________________________
# FIT SVM (Linear) - Cross Validation
svm_cv_default_results_up <- fit_resamples(
  svm_linear_wf_default_up,
  resamples = els_folds_up,
  metrics = test_metrics,
  control = control_resamples(save_pred = TRUE)
)

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

# View CV results
collect_metrics(svm_cv_default_results_up)




# Fit final model on full training data
svm_final_default_up <- fit(svm_linear_wf_default_up, data = els_train_up)

# Save untuned model
saveRDS(svm_cv_default_results_up, "svm_cv_default_results_up.rds")
saveRDS(svm_final_default_up, "svm_final_default_up.rds")



#____________________________
# MAKE PREDICTIONS ON TEST SET
svm_test_predictions_default_up <- predict(svm_final_default_up, els_test_clean, type = "prob") |>
  bind_cols(predict(svm_final_default_up, els_test_clean, type = "class")) |>
  bind_cols(els_test_clean |> select(dropout_factor, case_wts_rounded))

#____________________________
# EVALUATE TEST SET PERFORMANCE
svm_test_results_default_up <- svm_test_predictions_default_up |>
  test_metrics(truth = dropout_factor,
               estimate = .pred_class,
               .pred_Dropout,
               case_weights = case_wts_rounded)

print("SVM Linear Test Set Performance (Untuned, Upsampled):")
print(svm_test_results_default_up)

# Save test predictions
saveRDS(svm_test_predictions_default_up, "svm_test_predictions_default_up.rds")

#____________________________
# THRESHOLD SWEEP
threshold_grid <- tibble::tibble(threshold = seq(0.05, 0.5, by = 0.01))

svm_threshold_eval_default_up <- threshold_grid %>%
  mutate(
    metrics = map(
      threshold,
      ~ {
        svm_preds <- svm_test_predictions_default_up %>%
          mutate(
            .pred_class = factor(
              ifelse(.pred_Dropout >= .x, "Dropout", "Not_Dropout"),
              levels = levels(dropout_factor)
            )
          )
        test_metrics(
          svm_preds,
          truth        = dropout_factor,
          estimate     = .pred_class,
          .pred_Dropout,
          case_weights = case_wts_rounded
        )
      }
    )
  ) %>%
  unnest(metrics)

#____________________________
# VIEW THRESHOLD SWEEP RESULTS
svm_threshold_eval_default_up %>%
  filter(.metric %in% c("sens", "spec", "f_meas")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(across(c(sens, spec, f_meas), ~ round(.x, 3))) %>%
  print(n = Inf)

# Save threshold results
saveRDS(svm_threshold_eval_default_up, "svm_threshold_eval_default_up.rds")

#_______________________________________________________________________________






















#_______________________________________________________________________________
#_______________________________________________________________________________
##### Support Vector Machine LINEAR (Tuned, Upsampled) - UNWEIGHTED ####
#_______________________________________________________________________________
# NOTE: SVM results are UNWEIGHTED due to kernlab/LiblineaR not supporting 
# case weights in tidymodels.
#_______________________________________________________________________________

# Load required libraries
library(kernlab)
library(doParallel)
library(parallel)

# Set up parallel processing
cl <- makeCluster(10)
registerDoParallel(cl)

#_______________________________________________________________________________
# SVM Linear with kernlab (NO CASE WEIGHTS)
#_______________________________________________________________________________

# SVM Linear Model specification (TUNED, unweighted)
svm_linear_spec_unweighted <- svm_linear(
  cost = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# Workflow WITHOUT case weights
svm_linear_wf_unweighted <- workflow() %>%
  # NO add_case_weights() - kernlab doesn't support it
  add_formula(
    dropout_factor ~ BYS20A + BYS20B + BYS20C + BYS20D + BYS20E + 
      BYS20F + BYS20G + BYS20H + BYS20I + BYS20J + BYS20K + 
      BYS20L + BYS20M + BYS20N + BYS21A + BYS21B + BYS21C + 
      BYS21D + BYS21E + BYS22A + BYS22B + BYS22C + BYS22D + 
      BYS22E + BYS22G + BYS22H + BYS27H + bysex + byrace
  ) %>%
  add_model(svm_linear_spec_unweighted)

# Tuning grid (same cost range as before)
svm_grid <- grid_regular(
  cost(range = c(-1, -0.05), trans = log10_trans()),  # 0.1 to 0.891
  levels = 3
)

# Tune the model
svm_linear_tune_unweighted <- tune_grid(
  svm_linear_wf_unweighted,
  resamples = els_folds_up,
  grid = svm_grid,
  metrics = metric_set(accuracy, roc_auc, yardstick::sens, yardstick::spec, yardstick::f_meas),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

saveRDS(svm_linear_tune_unweighted, "svm_linear_tune_unweighted.rds")

#_______________________________________________________________________________
# View results
#_______________________________________________________________________________

show_best(svm_linear_tune_unweighted, metric = "roc_auc", n = 4)
show_best(svm_linear_tune_unweighted, metric = "sens", n = 4)

# Get ALL metrics for best model
select_best(svm_linear_tune_unweighted, metric = "roc_auc") %>%
  inner_join(collect_metrics(svm_linear_tune_unweighted), by = ".config") %>%
  select(.metric, mean, std_err) %>%
  mutate(across(c(mean, std_err), ~ round(.x, 3)))









#_______________________________________________________________________________

#_______________________________________________________________________________

#_______________________________________________________________________________

#_______________________________________________________________________________
#_______________________________________________________________________________
##### Support Vector Machine RBF (Tuned, Upsampled) - UNWEIGHTED ####
#_______________________________________________________________________________
# NOTE: SVM results are UNWEIGHTED due to kernlab not supporting 
# case weights in tidymodels.
#_______________________________________________________________________________

# Load required libraries
library(kernlab)
library(doParallel)
library(parallel)

# Set up parallel processing
cl <- makeCluster(10)
registerDoParallel(cl)

#_______________________________________________________________________________
# SVM RBF with kernlab (NO CASE WEIGHTS)
#_______________________________________________________________________________

# SVM RBF Model specification (TUNED, unweighted)
svm_rbf_spec_unweighted <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# Workflow WITHOUT case weights
svm_rbf_wf_unweighted <- workflow() %>%
  # NO add_case_weights() - kernlab doesn't support it
  add_formula(
    dropout_factor ~ BYS20A + BYS20B + BYS20C + BYS20D + BYS20E + 
      BYS20F + BYS20G + BYS20H + BYS20I + BYS20J + BYS20K + 
      BYS20L + BYS20M + BYS20N + BYS21A + BYS21B + BYS21C + 
      BYS21D + BYS21E + BYS22A + BYS22B + BYS22C + BYS22D + 
      BYS22E + BYS22G + BYS22H + BYS27H + bysex + byrace
  ) %>%
  add_model(svm_rbf_spec_unweighted)

# Tuning grid (cost + rbf_sigma)
svm_rbf_grid <- grid_regular(
  cost(range = c(-1, -0.05), trans = log10_trans()),      # 0.1 to 0.891
  rbf_sigma(range = c(-3, -2), trans = log10_trans()),    # 0.001 to 0.01 (conservative)
  levels = 3  # 3x3 = 9 combinations
)

# Check what values you're testing
print(svm_rbf_grid)

# Tune the model
svm_rbf_tune_unweighted <- tune_grid(
  svm_rbf_wf_unweighted,
  resamples = els_folds_up,
  grid = svm_rbf_grid,
  metrics = metric_set(accuracy, roc_auc, yardstick::sens, yardstick::spec, yardstick::f_meas),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

#_______________________________________________________________________________
# SAVE RESULTS IMMEDIATELY
#_______________________________________________________________________________

saveRDS(svm_rbf_tune_unweighted, "svm_rbf_tune_unweighted.rds")
saveRDS(svm_rbf_wf_unweighted, "svm_rbf_wf_unweighted.rds")
saveRDS(svm_rbf_grid, "svm_rbf_grid.rds")



#_______________________________________________________________________________
# View results
#_______________________________________________________________________________

show_best(svm_rbf_tune_unweighted, metric = "roc_auc", n = 5)
show_best(svm_rbf_tune_unweighted, metric = "sens", n = 5)

# Get ALL metrics for best model
select_best(svm_rbf_tune_unweighted, metric = "roc_auc") %>%
  inner_join(collect_metrics(svm_rbf_tune_unweighted), by = ".config") %>%
  select(.metric, mean, std_err) %>%
  mutate(across(c(mean, std_err), ~ round(.x, 3)))


#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________
##### SVM RBF - TRAIN ON FULL DATA & TEST SET EVALUATION ####
#_______________________________________________________________________________

# 1. Select best hyperparameters
svm_rbf_best_params <- select_best(svm_rbf_tune_unweighted, metric = "roc_auc")
# "Best hyperparameters:")
print(svm_rbf_best_params)

# 2. Finalize workflow with best parameters
svm_rbf_final_workflow <- finalize_workflow(svm_rbf_wf_unweighted, svm_rbf_best_params)

# 3. Fit final model on FULL training data
svm_rbf_final_fit <- fit(svm_rbf_final_workflow, data = els_train_up)

# 4. Save final model
saveRDS(svm_rbf_final_fit, "svm_rbf_final_fit_unweighted.rds")

# 5. Make predictions on test set
svm_rbf_test_predictions <- predict(svm_rbf_final_fit, els_test_clean, type = "prob") %>%
  bind_cols(predict(svm_rbf_final_fit, els_test_clean, type = "class")) %>%
  bind_cols(els_test_clean %>% select(dropout_factor, case_wts_rounded))

# 6. Save test predictions
saveRDS(svm_rbf_test_predictions, "svm_rbf_test_predictions.rds")

#_______________________________________________________________________________

#_______________________________________________________________________________
##### TEST SET PERFORMANCE - SVM RBF ####
#_______________________________________________________________________________

# Calculate test metrics (UNWEIGHTED - SVM doesn't support case weights)
svm_rbf_test_metrics <- metric_set(
  accuracy, roc_auc, yardstick::sens, yardstick::spec, yardstick::f_meas
)(
  svm_rbf_test_predictions,
  truth = dropout_factor,
  estimate = .pred_class,
  .pred_Dropout
)

print("=== SVM RBF Test Set Performance (UNWEIGHTED) ===")
print(svm_rbf_test_metrics)


#_______________________________________________________________________________
##### THRESHOLD SWEEP - SVM RBF ####
#_______________________________________________________________________________

threshold_grid <- tibble::tibble(threshold = seq(0.05, 0.5, by = 0.01))

svm_rbf_threshold_eval <- threshold_grid %>%
  mutate(
    metrics = map(
      threshold,
      ~ {
        svm_rbf_preds <- svm_rbf_test_predictions %>%
          mutate(
            .pred_class = factor(
              ifelse(.pred_Dropout >= .x, "Dropout", "Not_Dropout"),
              levels = levels(dropout_factor)
            )
          )
        metric_set(accuracy, roc_auc, yardstick::sens, yardstick::spec, yardstick::f_meas)(
          svm_rbf_preds,
          truth = dropout_factor,
          estimate = .pred_class,
          .pred_Dropout
          # NO case_weights - SVM is unweighted
        )
      }
    )
  ) %>%
  unnest(metrics)

#_______________________________________________________________________________
# VIEW THRESHOLD SWEEP RESULTS
#_______________________________________________________________________________

svm_rbf_threshold_eval %>%
  filter(.metric %in% c("sens", "spec", "f_meas")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(across(c(sens, spec, f_meas), ~ round(.x, 3))) %>%
  print(n = Inf)

# Save threshold results
saveRDS(svm_rbf_threshold_eval, "svm_rbf_threshold_eval.rds")






#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#_______________________________________________________________________________
#_______________________________________________________________________________
##### XGBoost - TRUE DEFAULT Parameters (Completely Untuned) ####
#_______________________________________________________________________________

# Load required library
library(xgboost)  # For XGBoost engine

# Set up parallel processing
library(doParallel)
library(parallel)

cl <- makeCluster(10)
registerDoParallel(cl)

#_______________________________________________________________________________
# XGBOOST - TRUE DEFAULT MODEL (NO CUSTOM PARAMETERS) - UPSAMPLED
#_______________________________________________________________________________

# XGBoost Model specification (using XGBoost's built-in defaults)
xgb_spec_true_default <- boost_tree() %>%  # NO parameters specified at all!
  set_engine("xgboost") %>%                 # NO engine arguments!
  set_mode("classification")

# XGBoost Workflow with case weights
xgb_workflow_true_default <- workflow() %>%
  add_case_weights(case_wts_rounded) %>%
  add_formula(
    dropout_factor ~ BYS20A + BYS20B + BYS20C + BYS20D + BYS20E + 
      BYS20F + BYS20G + BYS20H + BYS20I + BYS20J + BYS20K + 
      BYS20L + BYS20M + BYS20N + BYS21A + BYS21B + BYS21C + 
      BYS21D + BYS21E + BYS22A + BYS22B + BYS22C + BYS22D + 
      BYS22E + BYS22G + BYS22H + BYS27H + bysex + byrace
  ) %>%
  add_model(xgb_spec_true_default)

#____________________________
# FIT XGBOOST (True Defaults) - Cross Validation
#____________________________

xgb_cv_true_default_results <- fit_resamples(
  xgb_workflow_true_default,
  resamples = els_folds_up,  # 5-fold CV on upsampled data
  metrics = metric_set(accuracy, roc_auc, yardstick::sens, yardstick::spec, yardstick::f_meas),
  control = control_resamples(save_pred = TRUE)
)

# Clean up parallel processing
stopCluster(cl)
registerDoSEQ()

#____________________________
# SAVE CV results
#____________________________

saveRDS(xgb_cv_true_default_results, "xgb_cv_true_default_results.rds")
saveRDS(xgb_workflow_true_default, "xgb_workflow_true_default.rds")

#____________________________
# VIEW RESULTS
#____________________________

collect_metrics(xgb_cv_true_default_results)


#_______________________________________________________________________________
##### XGBOOST TRUE DEFAULT - TRAIN ON FULL DATA & TEST SET EVALUATION ####
#_______________________________________________________________________________

# 1. Fit final model on FULL training data
xgb_true_default_final_fit <- fit(xgb_workflow_true_default, data = els_train_up)

# 2. Save final model
saveRDS(xgb_true_default_final_fit, "xgb_true_default_final_fit.rds")

# 3. Make predictions on test set
xgb_true_default_test_predictions <- predict(xgb_true_default_final_fit, els_test_clean, type = "prob") %>%
  bind_cols(predict(xgb_true_default_final_fit, els_test_clean, type = "class")) %>%
  bind_cols(els_test_clean %>% select(dropout_factor, case_wts_rounded))

# 4. Save test predictions
saveRDS(xgb_true_default_test_predictions, "xgb_true_default_test_predictions.rds")

#_______________________________________________________________________________
##### TEST SET PERFORMANCE ####
#_______________________________________________________________________________

# Calculate weighted test metrics
xgb_true_default_test_results <- metric_set(
  accuracy, roc_auc, yardstick::sens, yardstick::spec, yardstick::f_meas
)(
  xgb_true_default_test_predictions,
  truth = dropout_factor,
  estimate = .pred_class,
  .pred_Dropout,
  case_weights = case_wts_rounded
)


print(xgb_true_default_test_results)



#_______________________________________________________________________________
##### THRESHOLD SWEEP ####
#_______________________________________________________________________________

threshold_grid <- tibble::tibble(threshold = seq(0.05, 0.5, by = 0.01))

xgb_true_default_threshold_eval <- threshold_grid %>%
  mutate(
    metrics = map(
      threshold,
      ~ {
        xgb_preds <- xgb_true_default_test_predictions %>%
          mutate(
            .pred_class = factor(
              ifelse(.pred_Dropout >= .x, "Dropout", "Not_Dropout"),
              levels = levels(dropout_factor)
            )
          )
        metric_set(accuracy, roc_auc, yardstick::sens, yardstick::spec, yardstick::f_meas)(
          xgb_preds,
          truth = dropout_factor,
          estimate = .pred_class,
          .pred_Dropout,
          case_weights = case_wts_rounded
        )
      }
    )
  ) %>%
  unnest(metrics)

#_______________________________________________________________________________
# VIEW THRESHOLD SWEEP RESULTS
#_______________________________________________________________________________

xgb_true_default_threshold_eval %>%
  filter(.metric %in% c("sens", "spec", "f_meas")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(across(c(sens, spec, f_meas), ~ round(.x, 3))) %>%
  print(n = Inf)

# Save threshold results
saveRDS(xgb_true_default_threshold_eval, "xgb_true_default_threshold_eval.rds")

message("\n✓ XGBoost True Default test evaluation complete!")





#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>











#_______________________________________________________________________________
#####XGBoost - Tuned Model (Grid Search)####
#_______________________________________________________________________________

# Load required library
library(xgboost)

# Set up parallel processing
library(doParallel)
library(parallel)

cl <- makeCluster(10)
registerDoParallel(cl)

#_______________________________________________________________________________
# XGBOOST - TUNABLE MODEL SPECIFICATION - UPSAMPLED
#_______________________________________________________________________________

# XGBoost Model specification - CONSERVATIVE (Anti-Overfitting)
xgb_spec_tuned_up <- boost_tree(
  trees = 300,              # ✓ Reduced from 500
  tree_depth = tune(),      # Will use 1-3 range
  learn_rate = tune(),      # Will use 0.001-0.01 range  
  min_n = tune(),           # Will use 100-200 range
  mtry = 10,                # ✓ Reduced from 15
  loss_reduction = 1,       # ✓ Increased from 0.01 (harder to split)
  sample_size = 0.6         # ✓ Reduced from 0.8 (more randomness)
) %>%
  set_engine("xgboost", 
             
             #   - L1 eliminates weak features, L2 shrinks strong features
             #   - Together they create "Elastic Net" style regularization
             stop_iter = 15,        #  Reduced from 20
             reg_alpha = 0.5,       # L1 Lasso like penalty
             reg_lambda = 0.5       # L2 Ridge Like
  ) %>%
  set_mode("classification")
#_______________________________________________________________________________
# CREATE TUNING GRID - CONSERVATIVE (ANTI-OVERFITTING)
#_______________________________________________________________________________

# Conservative grid designed to prevent overfitting to duplicate students
xgb_grid <- grid_regular(
  
  # TREE_DEPTH: Controls how deep each decision tree can grow
  # OLD Range: 2 to 6 → NEW Range: 1 to 3 (MUCH shallower)
  # Testing: Very shallow (1) vs shallow (2) vs moderate (3)
  # Why changed:
  #   - Your data showed depth=6 models had 99.5% CV but 59.8% test (massive overfit!)
  #   - Deep trees (4-6) were memorizing exact combinations of duplicate students
  #   - Shallow trees (1-3) CANNOT memorize complex patterns, forced to generalize
  # Specific effects by depth:
  #   - Depth=1: Only 1 split per tree (stump) = extremely simple, captures main effects only
  #   - Depth=2: Up to 3 terminal nodes = can capture simple interactions
  #   - Depth=3: Up to 7 terminal nodes = moderate complexity, still generalizes well
  # This is the MOST IMPORTANT change to prevent overfitting with duplicates
  tree_depth(range = c(1, 3)),
  
  # LEARN_RATE: How much each tree contributes to the final prediction
  # OLD Range: 10^-2.5 to 10^-1 (0.00316 to 0.1) → NEW: 10^-3 to 10^-2 (0.001 to 0.01)
  # Testing: Very slow (0.001) vs slow (0.00316) vs moderate (0.01)
  # Why changed:
  #   - Fast learning (0.1) was top performer on CV but failed on test
  #   - High learn_rate makes aggressive updates that fit training noise perfectly
  #   - Slow learning makes small, careful adjustments = more stable, generalizes better
  # Specific effects by rate:
  #   - 0.001: Very gradual learning, needs many trees, most conservative
  #   - 0.00316: Slow but practical, good balance of speed and stability  
  #   - 0.01: Moderate pace, still conservative compared to 0.1
  # Trade-off: Slower learning takes longer to train but much less overfitting
  # Note: trans = log10_trans() spaces values logarithmically (0.001, 0.00316, 0.01)
  learn_rate(range = c(-3, -2), trans = log10_trans()),
  
  # MIN_N: Minimum number of observations required in a node to keep splitting
  # OLD Range: 20 to 40 → NEW Range: 100 to 200 (5-10x LARGER)
  # Testing: Large nodes (100) vs larger (150) vs largest (200)
  # Why changed:
  #   - Small min_n (20-40) allows tree to split on tiny groups
  #   - With duplicates, model finds "Student #42 appears 15 times with these exact values"
  #   - Large min_n (100-200) requires 100-200 students to make a split
  #   - IMPOSSIBLE to memorize individual duplicates with such large groups
  # Specific effects by min_n:
  #   - 100: Must have 100+ students to split = forced to find broad patterns
  #   - 150: Even more conservative, only very strong patterns survive
  #   - 200: Maximum regularization, captures only the most general relationships
  # This is CRITICAL with upsampled data - prevents memorization of duplicate groups
  # Example: With min_n=200, can't create rule "if BYS20A=3 AND BYS20B=2 AND byrace=White"
  #          because that specific combo (even with 15 duplicates) < 200 students
  min_n(range = c(100, 200)),
  
  # LEVELS: How many values to test for EACH parameter above
  # With 3 parameters and levels=3: tests 3 × 3 × 3 = 27 combinations
  # Each parameter gets 3 evenly-spaced values within its range:
  #   - tree_depth: 1, 2, 3
  #   - learn_rate: 0.001, 0.00316, 0.01
  #   - min_n: 100, 150, 200
  levels = 3
)



# View the full grid
print(xgb_grid, n = Inf)
#_______________________________________________________________________________
# CREATE WORKFLOW
#_______________________________________________________________________________

# XGBoost Workflow with case weights
xgb_workflow_tuned_up <- workflow() %>%
  add_case_weights(case_wts_rounded) %>%
  add_formula(
    dropout_factor ~ BYS20A + BYS20B + BYS20C + BYS20D + BYS20E + 
      BYS20F + BYS20G + BYS20H + BYS20I + BYS20J + BYS20K + 
      BYS20L + BYS20M + BYS20N + BYS21A + BYS21B + BYS21C + 
      BYS21D + BYS21E + BYS22A + BYS22B + BYS22C + BYS22D + 
      BYS22E + BYS22G + BYS22H + BYS27H + bysex + byrace
  ) %>%
  add_model(xgb_spec_tuned_up)

#_______________________________________________________________________________
# TUNE XGBOOST - Grid Search with Cross-Validation
#_______________________________________________________________________________

xgb_tune_results_up <- tune_grid(
  xgb_workflow_tuned_up,
  resamples = els_folds_up,
  grid = xgb_grid,
  metrics = test_metrics,
  control = control_grid(
    save_pred = TRUE,
    verbose = TRUE
  )
)
# Clean up parallel processing
stopCluster(cl)
registerDoSEQ()
#_______________________________________________________________________________
# SAVE TUNING RESULTS
#_______________________________________________________________________________

saveRDS(xgb_tune_results_up, "xgb_tune_results_up.rds")
saveRDS(xgb_grid, "xgb_grid.rds")
saveRDS(xgb_workflow_tuned_up, "xgb_workflow_tuned_up.rds")

# Load back later
xgb_tune_results_up <- readRDS("xgb_tune_results_up.rds")

#_______________________________________________________________________________
# VIEW TUNING RESULTS
#_______________________________________________________________________________

collect_metrics(xgb_tune_results_up)



#_______________________________________________________________________________
# 2. SEE BEST PERFORMING CONFIGURATIONS
#_______________________________________________________________________________

# Best for sensitivity (catching dropouts)
show_best(xgb_tune_results_up, metric = "sens", n = 5)

# Best overall (ROC AUC)
show_best(xgb_tune_results_up, metric = "roc_auc", n = 5)

# Best F1 score
show_best(xgb_tune_results_up, metric = "f_meas", n = 5)

# Select best by ROC AUC
xgb_best_params <- select_best(xgb_tune_results_up, metric = "roc_auc")
xgb_best_params


# Method 1: Using the .config identifier (BEST METHOD)
collect_metrics(xgb_tune_results_up) %>%
  filter(.config == "Preprocessor1_Model09") %>%
  select(.metric, mean, std_err, n) %>%
  mutate(mean = round(mean, 3), std_err = round(std_err, 4))

#_______________________________________________________________________________


#_______________________________________________________________________________
##### XGBOOST - TRAIN FINAL MODEL & TEST SET EVALUATION ####
#_______________________________________________________________________________

# Load if needed
xgb_tune_results_up <- readRDS("xgb_tune_results_up.rds")
xgb_workflow_tuned_up <- readRDS("xgb_workflow_tuned_up.rds")

#_______________________________________________________________________________
# 1. SELECT BEST HYPERPARAMETERS
#_______________________________________________________________________________

xgb_best_params_up <- select_best(xgb_tune_results_up, metric = "roc_auc")
cat("Best hyperparameters:\n")
print(xgb_best_params_up)

#_______________________________________________________________________________
# 2. FINALIZE WORKFLOW WITH BEST PARAMETERS
#_______________________________________________________________________________

xgb_final_workflow_up <- finalize_workflow(xgb_workflow_tuned_up, xgb_best_params_up)

#_______________________________________________________________________________
# 3. FIT FINAL MODEL ON FULL TRAINING DATA
#_______________________________________________________________________________

cat("\nFitting final XGBoost model on full training data...\n")
xgb_final_fit_up <- fit(xgb_final_workflow_up, data = els_train_up)

# Save final model
saveRDS(xgb_final_fit_up, "xgb_final_fit_up.rds")
saveRDS(xgb_best_params_up, "xgb_best_params_up.rds")



# To reload later:
xgb_final_fit_up <- readRDS("xgb_final_fit_up.rds")

#_______________________________________________________________________________
# 4. MAKE PREDICTIONS ON TEST SET
#_______________________________________________________________________________

cat("Making predictions on test set...\n")
xgb_test_predictions_up <- predict(xgb_final_fit_up, els_test_clean, type = "prob") |>
  bind_cols(predict(xgb_final_fit_up, els_test_clean, type = "class")) |>
  bind_cols(els_test_clean |> select(dropout_factor, case_wts_rounded))

# View structure of predictions
glimpse(xgb_test_predictions_up)

# Save test predictions
saveRDS(xgb_test_predictions_up, "xgb_test_predictions_up.rds")

#_______________________________________________________________________________
# 5. EVALUATE TEST SET PERFORMANCE
#_______________________________________________________________________________

cat("\n=== XGBoost Test Set Performance (Tuned, Upsampled) ===\n")

xgb_test_results_up <- xgb_test_predictions_up |>
  test_metrics(truth = dropout_factor,
               estimate = .pred_class,
               .pred_Dropout,
               case_weights = case_wts_rounded)

print(xgb_test_results_up)

# Save test results
saveRDS(xgb_test_results_up, "xgb_test_results_up.rds")


#_______________________________________________________________________________
# 7. THRESHOLD SWEEP
#_______________________________________________________________________________

cat("\n=== Running Threshold Sweep ===\n")

threshold_grid <- tibble::tibble(threshold = seq(0.05, 0.5, by = 0.01))

xgb_threshold_eval_up <- threshold_grid %>%
  mutate(
    metrics = map(
      threshold,
      ~ {
        xgb_preds <- xgb_test_predictions_up %>%
          mutate(
            .pred_class = factor(
              ifelse(.pred_Dropout >= .x, "Dropout", "Not_Dropout"),
              levels = levels(dropout_factor)
            )
          )
        test_metrics(
          xgb_preds,
          truth        = dropout_factor,
          estimate     = .pred_class,
          .pred_Dropout,
          case_weights = case_wts_rounded
        )
      }
    )
  ) %>%
  unnest(metrics)

#_______________________________________________________________________________
# 8. VIEW THRESHOLD SWEEP RESULTS
#_______________________________________________________________________________

cat("\n=== Threshold Sweep Results ===\n")
xgb_threshold_eval_up %>%
  filter(.metric %in% c("sens", "spec", "f_meas")) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  mutate(across(c(sens, spec, f_meas), ~ round(.x, 3))) %>%
  print(n = Inf)

# Save threshold results
saveRDS(xgb_threshold_eval_up, "xgb_threshold_eval_up.rds")


#_______________________________________________________________________________
# 10. VISUALIZE THRESHOLD TRADEOFFS
#_______________________________________________________________________________

library(ggplot2)

# Plot sens/spec tradeoff
xgb_threshold_eval_up %>%
  filter(.metric %in% c("sens", "spec")) %>%
  ggplot(aes(x = threshold, y = .estimate, color = .metric)) +
  geom_line(size = 1) +
  geom_vline(xintercept = 0.5, linetype = "dashed", alpha = 0.5) +
  labs(title = "XGBoost: Sensitivity vs Specificity by Threshold",
       x = "Probability Threshold",
       y = "Performance",
       color = "Metric") +
  theme_minimal() +
  scale_color_manual(values = c("sens" = "blue", "spec" = "red"),
                     labels = c("Sensitivity (Dropout)", "Specificity (Not Dropout)"))

# Plot F1 score
xgb_threshold_eval_up %>%
  filter(.metric == "f_meas") %>%
  ggplot(aes(x = threshold, y = .estimate)) +
  geom_line(size = 1, color = "darkgreen") +
  geom_vline(xintercept = 0.5, linetype = "dashed", alpha = 0.5) +
  labs(title = "XGBoost: F1 Score by Threshold",
       x = "Probability Threshold",
       y = "F1 Score") +
  theme_minimal()

cat("\n✓ XGBoost evaluation complete!\n")
#_______________________________________________________________________________







#_______________________________________________________________________________
##### LOAD BEST PERFORMING XGBOOST MODEL ####
#_______________________________________________________________________________

# Load the final fitted model (trained on full training data)
xgb_final_fit_up <- readRDS("xgb_final_fit_up.rds")

# Load the best hyperparameters
xgb_best_params_up <- readRDS("xgb_best_params_up.rds")

# Load test predictions (this exists!)
xgb_test_predictions_up <- readRDS("xgb_test_predictions_up.rds")

# Load threshold sweep results
xgb_threshold_eval_up <- readRDS("xgb_threshold_eval_up.rds")

# Load tuning results (to see CV performance)
xgb_tune_results_up <- readRDS("xgb_tune_results_up.rds")

# load test reseults
xgb_final_fit_up <- readRDS("xgb_final_fit_up.rds")












#_______________________________________________________________________________

#_______________________________________________________________________________
# predict and save the best model
#_______________________________________________________________________________

#Choose the best performing model. save it w. adjusted threshold:
# Create a custom prediction function with 0.40 threshold
predict_with_threshold <- function(model, new_data, threshold = 0.40) {
  # Get probability predictions
  probs <- predict(model, new_data, type = "prob")
  
  # Apply custom threshold
  predictions <- ifelse(probs$.pred_Dropout >= threshold, "Dropout", "Not_Dropout")
  
  # Return in the same format as type = "class" predictions
  tibble(.pred_class = factor(predictions, levels = c("Dropout", "Not_Dropout")))
}

# Save the custom prediction function along with the model
penalized_model_with_threshold <- list(
  model = penalized_model_fully_trained_up,
  predict_func = predict_with_threshold,
  threshold = 0.40
)

saveRDS(penalized_model_with_threshold, "penalized_model_fully_trained_up_with_threshold.rds")




#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>
#>

#_______________________________________________________________________________
##### VARIABLE IMPORTANCE PLOT (VIP NEW) ####
#_______________________________________________________________________________


library(vip)
library(ggplot2)

# Load model
xgb_final_fit_up <- readRDS("xgb_final_fit_up.rds")
xgb_fitted_model <- extract_fit_parsnip(xgb_final_fit_up)

# Get VIP data for TOP 5 (or whatever number you want)
vip_data_top5 <- vip(xgb_fitted_model, num_features = 5)$data

# Create variable labels mapping (same as before)
variable_labels <- c(
  "BYS22D.L" = "Got in a fight",
  "BYS22B.L" = "Offered drugs at school",
  "BYS20N.L" = "Racial fights occur",
  "byraceWhite" = "Race: White",
  "BYS22D.Q" = "Got in a fight (quadratic)",
  "BYS20M.L" = "Gangs in school",
  "BYS20J.L" = "Don't feel safe",
  "byraceBlack" = "Race: Black",
  "BYS20AAgree" = "Students get along with teachers: Agree",
  "BYS27H.L" = "Teachers expect excellence"
)

# Add readable labels
vip_data_top5$Variable_Label <- ifelse(vip_data_top5$Variable %in% names(variable_labels),
                                       variable_labels[vip_data_top5$Variable],
                                       vip_data_top5$Variable)

# Create NEW plot with top 5
vip_plot_top5 <- ggplot(vip_data_top5, aes(x = Importance, y = reorder(Variable_Label, Importance))) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  labs(x = "Importance", y = NULL) +
  theme_minimal()

# Adjust formatting
vip_plot_top5_adjusted <- vip_plot_top5 + 
  scale_x_continuous(
    breaks = seq(0, 0.15, by = 0.01)
  ) + 
  theme(
    axis.text.y = element_text(size = 14),
    axis.text.x = element_text(size = 14, angle = 45, hjust = 1),
    axis.title.x = element_text(size = 13),
    plot.title = element_blank(),
    plot.subtitle = element_blank()
  )

# Display
print(vip_plot_top5_adjusted)

# Save
saveRDS(vip_plot_top5_adjusted, "vip_plot_top5.rds")
ggsave("vip_plot_top5.png", vip_plot_top5_adjusted, 
       width = 10, height = 5, dpi = 300)









# Load the saved plot
vip_plot <- readRDS("vip_plot.rds")

# Adjust font sizes and x-axis
vip_plot_adjusted <- vip_plot + 
  scale_x_continuous(
    breaks = seq(0, 0.15, by = 0.01)  # Every 0.01 from 0 to 0.15
  ) + 
  theme(
    axis.text.y = element_text(size = 14),        # Y-axis labels (variable names)
    axis.text.x = element_text(size = 14, angle = 45, hjust = 1),  # X-axis numbers, rotated
    axis.title.x = element_text(size = 13),       # X-axis title "Importance"
    plot.title = element_blank(),                 # Remove title
    plot.subtitle = element_blank()               # Remove subtitle
  )

# Display the plot
print(vip_plot_adjusted)








library(vip)
library(xgboost)

# Load your model
xgb_final_fit_up <- readRDS("xgb_final_fit_up.rds")
xgb_fitted_model <- extract_fit_parsnip(xgb_final_fit_up)

# Get importance matrix with exact values
xgb_model <- xgb_fitted_model$fit
importance_matrix <- xgboost::xgb.importance(model = xgb_model)

# View all importance scores
print(importance_matrix)

# Get specific variables
importance_matrix %>%
  filter(Feature %in% c("BYS22D.L", "BYS22E.L", "BYS20N.L"))

# Or just the top 10
head(importance_matrix, 10)

# Save to CSV for easy reference
write.csv(importance_matrix, "xgb_importance_values.csv", row.names = FALSE)

# Get formatted output
importance_matrix %>%
  select(Feature, Gain) %>%
  mutate(Gain = round(Gain, 4)) %>%
  print(n = Inf)




library(vip)
library(ggplot2)
library(xgboost)

# Load model
xgb_final_fit_up <- readRDS("xgb_final_fit_up.rds")
xgb_fitted_model <- extract_fit_parsnip(xgb_final_fit_up)

# Get VIP data (TOP 10)
vip_data <- vip(xgb_fitted_model, num_features = 10)$data

# Check the actual variable names and importance values
print("Raw VIP data:")
print(vip_data)

# Get the full importance matrix to cross-check
xgb_model <- xgb_fitted_model$fit
importance_matrix <- xgboost::xgb.importance(model = xgb_model)

print("\nTop 10 from xgb.importance:")
print(head(importance_matrix, 10))

# Create variable labels mapping
variable_labels <- c(
  "BYS22D.L" = "Got in a fight",
  "BYS22B.L" = "Offered drugs at school",
  "BYS20N.L" = "Racial fights occur",
  "byraceWhite" = "Race: White",
  "BYS22D.Q" = "Got in a fight (quadratic)",
  "BYS20M.L" = "Gangs in school",
  "BYS20J.L" = "Don't feel safe",
  "byraceBlack" = "Race: Black",
  "BYS20AAgree" = "Students get along with teachers: Agree",
  "BYS27H.L" = "Teachers expect excellence",
  "BYS20E.L" = "Teaching is good",
  "BYS20ADisagree" = "Students get along with teachers: Disagree",
  "BYS20F.L" = "Teachers interested in students",
  "BYS21B.C" = "Rules are fair (cubic)",
  "BYS21B.L" = "Rules are fair",
  "BYS22E.L" = "Someone hit me",
  "BYS22E.Q" = "Someone hit me (quadratic)",
  "BYS22E.C" = "Someone hit me (cubic)"
)

# Add readable labels
vip_data$Variable_Label <- ifelse(vip_data$Variable %in% names(variable_labels),
                                  variable_labels[vip_data$Variable],
                                  vip_data$Variable)

# Print mapping to verify
print("\nVariable mapping:")
print(vip_data[, c("Variable", "Variable_Label", "Importance")])

# Create plot with readable labels
vip_plot <- ggplot(vip_data, aes(x = Importance, y = reorder(Variable_Label, Importance))) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  labs(title = "Variable Importance - XGBoost Model",
       subtitle = "Top 10 Predictors of Student Dropout",
       x = "Importance",
       y = NULL) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 11))

# Display the plot
print(vip_plot)

# Save outputs
saveRDS(vip_plot, "vip_plot.rds")
saveRDS(vip_data, "vip_data.rds")
write.csv(vip_data, "vip_data_with_labels.csv", row.names = FALSE)



#_______________________________________________________________________________
##### SHAP PLOT WITH READABLE LABELS - SAVE AS OBJECT ####
#_______________________________________________________________________________

library(SHAPforxgboost)
library(hardhat)

# Extract model
xgb_fitted_model <- extract_fit_parsnip(xgb_final_fit_up)
xgb_model <- xgb_fitted_model$fit

# Take a sample
set.seed(123)
train_sample <- els_train_up

# Preprocess to match model encoding
preprocessed <- hardhat::mold(
  xgb_final_fit_up$pre$actions$formula$formula,
  train_sample
)

train_matrix <- as.matrix(preprocessed$predictors)

# Add missing features as zeros (if any)
missing <- setdiff(xgb_model$feature_names, colnames(train_matrix))
if(length(missing) > 0) {
  full_matrix <- matrix(0, nrow = nrow(train_matrix), ncol = length(xgb_model$feature_names))
  colnames(full_matrix) <- xgb_model$feature_names
  full_matrix[, colnames(train_matrix)] <- train_matrix
  train_matrix <- full_matrix
}

# Run SHAP
shap_long <- shap.prep(xgb_model = xgb_model, X_train = train_matrix, top_n = 5)

# Create variable labels mapping
variable_labels <- c(
  "BYS22D.L" = "Got in a Fight",
  "BYS22B.L" = "Offered Drugs at School",
  "BYS20N.L" = "Racial Fights Occur",
  "byraceWhite" = "Race: White",
  "BYS22D.Q" = "Got in a Fight (Quadratic)"
)

# Replace variable names with readable labels
shap_long$variable <- factor(
  ifelse(shap_long$variable %in% names(variable_labels),
         variable_labels[shap_long$variable],
         shap_long$variable),
  levels = variable_labels[unique(shap_long$variable)]
)


# Create base plot
shap_plot <- shap.plot.summary(shap_long)

# Modify the text size (layer 3 is GeomText with the mean values)
shap_plot$layers[[3]]$aes_params$size <- 5  # Change from 3 to 5



# Create SHAP plot with labels and formatting
# Add your theme customizations
shap_plot <- shap_plot +
  theme(
    axis.text.y = element_text(size = 14),        # Variable names
    axis.text.x = element_text(size = 14, angle = 30, hjust = 1),
    axis.title.x = element_text(size = 13),
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 12)
  )

# Display
print(shap_plot)

# Save
saveRDS(shap_plot, "shap_summary_plot_top5.rds")
ggsave("shap_summary_plot_top5.png", shap_plot, width = 10, height = 5, dpi = 300)

#_______________________________________________________________________________
##### ALE PLOTS FOR TOP 5 - DROPOUT ONLY (SAVE ALL COMPONENTS) ####
#_______________________________________________________________________________

library(iml)
library(ggplot2)
library(patchwork)

# Top 5 variables (use base names, not the encoded versions)
top_5_vars <- c("BYS22D", "BYS22B", "BYS20N", "byrace", "BYS22D")  # BYS22D appears twice (linear + quadratic)

# But for ALE, we only need unique base variables
top_5_vars_unique <- c("BYS22D", "BYS22B", "BYS20N", "byrace")

# Create readable labels
var_labels <- c(
  "BYS22D" = "Got in a Fight (L and Q)",
  "BYS22B" = "Offered Drugs at School",
  "BYS20N" = "Racial Fights Occur",
  "byrace" = "Student Race/Ethnicity"
)

# Create test explainer data
test_explainer_data <- els_test_clean %>%
  select(BYS20A, BYS20B, BYS20C, BYS20D, BYS20E, 
         BYS20F, BYS20G, BYS20H, BYS20I, BYS20J, BYS20K, 
         BYS20L, BYS20M, BYS20N, BYS21A, BYS21B, BYS21C, 
         BYS21D, BYS21E, BYS22A, BYS22B, BYS22C, BYS22D, 
         BYS22E, BYS22G, BYS22H, BYS27H, bysex, byrace)

# Create predictor object
predictor <- Predictor$new(
  model = xgb_final_fit_up,
  data = test_explainer_data,
  y = as.numeric(els_test_clean$dropout_factor == "Dropout"),
  predict.function = function(model, newdata) {
    predict(model, newdata, type = "prob")$.pred_Dropout
  }
)

# Save predictor object
saveRDS(predictor, "predictor_dropout.rds")

# Create individual ALE effect objects and plots
ale_effects_list <- list()
plot_list <- list()

for(var in top_5_vars_unique) {
  cat("Calculating ALE for", var, "...\n")
  
  # Create ALE effect
  ale_effect <- FeatureEffect$new(
    predictor = predictor,
    feature = var,
    method = "ale"
  )
  
  # Save the ALE effect object
  ale_effects_list[[var]] <- ale_effect
  
  # Create plot
  p <- plot(ale_effect) +
    labs(title = var_labels[var],
         y = "Effect on Dropout\nProbability",
         x = NULL) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      axis.title.y = element_text(size = 9)
    )
  
  plot_list[[var]] <- p
}

# Save ALE effects list
saveRDS(ale_effects_list, "ale_effects_list.rds")

# Save individual plots list
saveRDS(plot_list, "ale_plot_list.rds")

# Combine all plots (2x2 grid since we have 4 variables)
ale_combined_plot <- wrap_plots(plot_list, ncol = 2) +
  plot_annotation(
    theme = theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 10, hjust = 0)
    )
  )

# Save combined plot
saveRDS(ale_combined_plot, "ale_combined_plot.rds")

# Display the plot
print(ale_combined_plot)
















#_______________________________________________________________________________

#_______________________________________________________________________________
##### CREATE XGBOOST MODEL WITH 0.44 THRESHOLD FOR SHINY APP ####
#_______________________________________________________________________________

# Load the trained model
xgb_final_fit_up <- readRDS("xgb_final_fit_up.rds")

# Create a wrapper function that applies the 0.44 threshold
predict_with_threshold_044 <- function(model, new_data) {
  # Get probability predictions
  probs <- predict(model, new_data, type = "prob")
  
  # Get class predictions using 0.44 threshold
  class_preds <- ifelse(probs$.pred_Dropout >= 0.44, "Dropout", "Not_Dropout")
  class_preds <- factor(class_preds, levels = c("Dropout", "Not_Dropout"))
  
  # Return both probabilities and thresholded classes
  list(
    probabilities = probs,
    predictions = tibble(.pred_class = class_preds),
    threshold = 0.44
  )
}

# Create a model package with everything needed for Shiny
xgb_model_package <- list(
  model = xgb_final_fit_up,
  threshold = 0.44,
  predict_function = predict_with_threshold_044,
  # Include metadata
  best_params = readRDS("xgb_best_params_up.rds"),
  test_performance = readRDS("xgb_test_results_up.rds"),
  # Feature names for reference
  feature_names = c("BYS20A", "BYS20B", "BYS20C", "BYS20D", "BYS20E", 
                    "BYS20F", "BYS20G", "BYS20H", "BYS20I", "BYS20J", "BYS20K", 
                    "BYS20L", "BYS20M", "BYS20N", "BYS21A", "BYS21B", "BYS21C", 
                    "BYS21D", "BYS21E", "BYS22A", "BYS22B", "BYS22C", "BYS22D", 
                    "BYS22E", "BYS22G", "BYS22H", "BYS27H", "bysex", "byrace")
)

# Save the complete package for Shiny
saveRDS(xgb_model_package, "xgb_model_package_044.rds")

cat("✓ Model package saved with 0.44 threshold!\n")

#_______________________________________________________________________________
##### TEST THE WRAPPED MODEL ####
#_______________________________________________________________________________

# Test it on a few observations
test_sample <- els_test_clean[1:5, ]

# Use the wrapper function
results <- predict_with_threshold_044(xgb_final_fit_up, test_sample)

# View results
cat("\nTest predictions with 0.44 threshold:\n")
print(data.frame(
  Actual = test_sample$dropout_factor,
  Prob_Dropout = round(results$probabilities$.pred_Dropout, 3),
  Predicted = results$predictions$.pred_class
))

#_______________________________________________________________________________
##### HOW TO USE IN SHINY APP ####
#_______________________________________________________________________________

# In your Shiny app, load the model package:
# xgb_pkg <- readRDS("xgb_model_package_044.rds")
#
# Make predictions:
# results <- xgb_pkg$predict_function(xgb_pkg$model, new_student_data)
#
# Access predictions:
# dropout_probability <- results$probabilities$.pred_Dropout
# predicted_class <- results$predictions$.pred_class
#
# The threshold is: xgb_pkg$threshold



# Load the ALE effects list
ale_effects_list <- readRDS("ale_effects_list.rds")

# Get exact values for each variable
for(var in names(ale_effects_list)) {
  cat("\n=== ALE values for", var, "===\n")
  
  # Extract the results dataframe
  ale_data <- ale_effects_list[[var]]$results
  
  # Print the data
  print(ale_data)
  
  # Or more readable format
  cat("\nFormatted:\n")
  print(data.frame(
    Level = ale_data[[1]],  # Feature values (first column)
    ALE = round(ale_data$.ale, 4)  # ALE effects
  ))
}

# Or get specific variable
cat("\n=== Fighting (BYS22D) exact values ===\n")
fighting_ale <- ale_effects_list[["BYS22D"]]$results
print(fighting_ale)

cat("\n=== Drug Offers (BYS22B) exact values ===\n")
drugs_ale <- ale_effects_list[["BYS22B"]]$results
print(drugs_ale)

cat("\n=== Racial Conflicts (BYS20N) exact values ===\n")
racial_ale <- ale_effects_list[["BYS20N"]]$results
print(racial_ale)

cat("\n=== Race (byrace) exact values ===\n")
race_ale <- ale_effects_list[["byrace"]]$results
print(race_ale)
