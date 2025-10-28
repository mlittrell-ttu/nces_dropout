library(shiny)
library(bslib)
library(DT)
library(tidymodels)
library(xgboost)

# Load model with error checking
tryCatch({
  xgb_model_package <- readRDS("xgb_model_package_044.rds")
  xgb_model <- xgb_model_package$model
  predict_with_threshold_func <- xgb_model_package$predict_function
  model_threshold <- xgb_model_package$threshold
  cat("XGBoost model with", model_threshold, "threshold loaded successfully\n")
}, error = function(e) {
  stop("Failed to load model: ", e$message)
})

# Prediction function
predict_school_climate <- function(survey_data, race, gender) {
  # Define factor levels as used in training
  likert_levels <- c("Strongly_Agree", "Agree", "Disagree", "Strongly_Disagree")
  frequency_levels <- c("Never", "Once_or_Twice", "More_than_Twice")
  race_levels <- c("Am_Indian", "Asian_PI", "Black", "Hispanic_NoRace", "Hispanic_Race", "Multiracial", "White")
  
  # Map UI race values to training data race values
  race_mapping <- c(
    "American Indian or Alaska Native" = "Am_Indian",
    "Asian or Pacific Islander" = "Asian_PI", 
    "Black or African American" = "Black",
    "Hispanic or Latino (No Race)" = "Hispanic_NoRace",
    "Hispanic or Latino (Race Specified)" = "Hispanic_Race",
    "White" = "White",
    "Two or More Races" = "Multiracial"
  )
  
  # Map UI gender to training data format
  gender_mapping <- c("Male" = "Male", "Female" = "Female")
  
  new_data <- data.frame(
    # Likert variables (BYS20 series)
    BYS20A = factor(likert_levels[survey_data[1]], levels = likert_levels, ordered = TRUE),
    BYS20B = factor(likert_levels[survey_data[2]], levels = likert_levels, ordered = TRUE),
    BYS20C = factor(likert_levels[survey_data[3]], levels = likert_levels, ordered = TRUE),
    BYS20D = factor(likert_levels[survey_data[4]], levels = likert_levels, ordered = TRUE),
    BYS20E = factor(likert_levels[survey_data[5]], levels = likert_levels, ordered = TRUE),
    BYS20F = factor(likert_levels[survey_data[6]], levels = likert_levels, ordered = TRUE),
    BYS20G = factor(likert_levels[survey_data[7]], levels = likert_levels, ordered = TRUE),
    BYS20H = factor(likert_levels[survey_data[8]], levels = likert_levels, ordered = TRUE),
    BYS20I = factor(likert_levels[survey_data[9]], levels = likert_levels, ordered = TRUE),
    BYS20J = factor(likert_levels[survey_data[10]], levels = likert_levels, ordered = TRUE),
    BYS20K = factor(likert_levels[survey_data[11]], levels = likert_levels, ordered = TRUE),
    BYS20L = factor(likert_levels[survey_data[12]], levels = likert_levels, ordered = TRUE),
    BYS20M = factor(likert_levels[survey_data[13]], levels = likert_levels, ordered = TRUE),
    BYS20N = factor(likert_levels[survey_data[14]], levels = likert_levels, ordered = TRUE),
    # Likert variables (BYS21 series)
    BYS21A = factor(likert_levels[survey_data[15]], levels = likert_levels, ordered = TRUE),
    BYS21B = factor(likert_levels[survey_data[16]], levels = likert_levels, ordered = TRUE),
    BYS21C = factor(likert_levels[survey_data[17]], levels = likert_levels, ordered = TRUE),
    BYS21D = factor(likert_levels[survey_data[18]], levels = likert_levels, ordered = TRUE),
    BYS21E = factor(likert_levels[survey_data[19]], levels = likert_levels, ordered = TRUE),
    # Frequency variables (BYS22 series)
    BYS22A = factor(frequency_levels[survey_data[20]], levels = frequency_levels, ordered = TRUE),
    BYS22B = factor(frequency_levels[survey_data[21]], levels = frequency_levels, ordered = TRUE),
    BYS22C = factor(frequency_levels[survey_data[22]], levels = frequency_levels, ordered = TRUE),
    BYS22D = factor(frequency_levels[survey_data[23]], levels = frequency_levels, ordered = TRUE),
    BYS22E = factor(frequency_levels[survey_data[24]], levels = frequency_levels, ordered = TRUE),
    # BYS22F is not used in the XGBoost model - skipped
    BYS22G = factor(frequency_levels[survey_data[26]], levels = frequency_levels, ordered = TRUE),
    BYS22H = factor(frequency_levels[survey_data[27]], levels = frequency_levels, ordered = TRUE),
    # Likert variable (BYS27 series)
    BYS27H = factor(likert_levels[survey_data[28]], levels = likert_levels, ordered = TRUE),
    # Demographics
    bysex = factor(gender_mapping[gender], levels = c("Male", "Female")),
    byrace = factor(race_mapping[race], levels = race_levels)
  )
  
  tryCatch({
    result <- predict_with_threshold_func(xgb_model, new_data)
    return(as.character(result$predictions$.pred_class[1]))
  }, error = function(e) {
    return(paste("Prediction Error:", e$message))
  })
}

# Data storage
responses_data <- reactiveValues(data = data.frame(
  StudentID = character(),
  Prediction = character(),
  DateTime = character(),
  Race = character(),
  Gender = character(),
  Q1 = character(), Q2 = character(), Q3 = character(), Q4 = character(), Q5 = character(),
  Q6 = character(), Q7 = character(), Q8 = character(), Q9 = character(), Q10 = character(),
  Q11 = character(), Q12 = character(), Q13 = character(), Q14 = character(), Q15 = character(),
  Q16 = character(), Q17 = character(), Q18 = character(), Q19 = character(), Q20 = character(),
  Q21 = character(), Q22 = character(), Q23 = character(), Q24 = character(), Q25 = character(),
  Q26 = character(), Q27 = character(), Q28 = character(),
  stringsAsFactors = FALSE
))

# UI
ui <- page_navbar(
  title = "School Climate Survey",
  theme = bs_theme(bootswatch = "flatly"),
  
  # Student Tab
  nav_panel(
    title = "Student Survey",
    
    h3("School Climate Survey"),
    
    textInput("student_id", "Student ID:", placeholder = "Enter your student ID"),
    
    selectInput("race", "Race/Ethnicity:",
                choices = c("Select..." = "",
                            "American Indian or Alaska Native",
                            "Asian or Pacific Islander", 
                            "Black or African American",
                            "Hispanic or Latino (No Race)",
                            "Hispanic or Latino (Race Specified)",
                            "White",
                            "Two or More Races")),
    
    selectInput("gender", "Gender:",
                choices = c("Select..." = "",
                            "Male",
                            "Female")),
    
    h4("Rate your agreement with each statement about your school:"),

    fluidRow(
      column(6,
        selectInput("q1", "Students get along with teachers (BYS20A)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q2", "There is real school spirit (BYS20B)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q3", "Students make diverse friends (BYS20C)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q4", "Other students disrupt class (BYS20D)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q5", "The teaching is good (BYS20E)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q6", "Teachers are interested in students (BYS20F)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q7", "When I work hard, teachers praise me (BYS20G)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q8", "Teachers put me down (BYS20H)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q9", "Students put me down (BYS20I)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q10", "I don't feel safe at school (BYS20J)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q11", "Student disruptions get in the way of learning (BYS20K)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q12", "Student misbehavior gets in the way of learning (BYS20L)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q13", "There are gangs in school (BYS20M)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q14", "Racial fights often occur (BYS20N)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree"))
      ),
      column(6,
        selectInput("q15", "All students know school rules (BYS21A)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q16", "The rules are fair (BYS21B)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q17", "Punishment for breaking rules is same no matter by who (BYS21C)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q18", "School rules are strictly enforced (BYS21D)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q19", "If a rule is broken, students know what punishment will follow (BYS21E)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree")),
        selectInput("q20", "Something was stolen from me at school (BYS22A)",
                    choices = c("Select..." = "", "Never", "Once or Twice", "More than Twice")),
        selectInput("q21", "Someone offered to sell me drugs at school (BYS22B)",
                    choices = c("Select..." = "", "Never", "Once or Twice", "More than Twice")),
        selectInput("q22", "Someone threatened to hurt me at school (BYS22C)",
                    choices = c("Select..." = "", "Never", "Once or Twice", "More than Twice")),
        selectInput("q23", "I got in a fight at school (BYS22D)",
                    choices = c("Select..." = "", "Never", "Once or Twice", "More than Twice")),
        selectInput("q24", "Someone hit me at school (BYS22E)",
                    choices = c("Select..." = "", "Never", "Once or Twice", "More than Twice")),
        selectInput("q26", "Someone damaged my belongings at school (BYS22G)",
                    choices = c("Select..." = "", "Never", "Once or Twice", "More than Twice")),
        selectInput("q27", "Someone bullied me at school (BYS22H)",
                    choices = c("Select..." = "", "Never", "Once or Twice", "More than Twice")),
        selectInput("q28", "I go to school because teachers expect me to exceed  (BYS27H)",
                    choices = c("Select..." = "", "Strongly Agree", "Agree", "Disagree", "Strongly Disagree"))
      )
    ),
    
    br(),
    actionButton("submit_survey", "Submit Survey", class = "btn-success")
  ),
  
  # Admin Tab  
  nav_panel(
    title = "Admin Panel",
    h3("Dashboard Overview"),
    
    # Score cards row
    fluidRow(
      column(3,
        div(class = "card bg-primary text-white mb-3",
          div(class = "card-body",
            h4(class = "card-title", textOutput("total_responses")),
            p(class = "card-text", "Total Responses")
          )
        )
      ),
      column(3,
        div(class = "card bg-danger text-white mb-3",
          div(class = "card-body",
            h4(class = "card-title", textOutput("high_risk_count")),
            p(class = "card-text", "High Risk Students")
          )
        )
      ),
      column(3,
        div(class = "card bg-success text-white mb-3",
          div(class = "card-body",
            h4(class = "card-title", textOutput("low_risk_count")),
            p(class = "card-text", "Students Not at Risk")
          )
        )
      ),
      column(3,
        div(class = "card bg-warning text-dark mb-3",
          div(class = "card-body",
            h4(class = "card-title", textOutput("risk_percentage")),
            p(class = "card-text", "Students at Risk")
          )
        )
      )
    ),
    
    br(),
    fluidRow(
      column(6,
        h3("Survey Responses"),
        div(style = "margin-top: 10px;",
          fileInput("upload_csv", NULL,
                    accept = c(".csv"),
                    buttonLabel = "Load CSV",
                    placeholder = ""),
          downloadButton("download_template", "Download Template", class = "btn btn-secondary btn-sm")
        )
      ),
      column(6,
        div(style = "text-align: right; margin-top: 45px;",
          downloadButton("download_csv", "Download CSV", class = "btn btn-primary")
        )
      )
    ),
    DT::dataTableOutput("responses_table")
  )
)

# Server
server <- function(input, output, session) {
  
  observeEvent(input$submit_survey, {
    
    # Validate inputs (note: q25 removed as BYS22F is not in model)
    required_fields <- c("student_id", "race", "gender",
                         paste0("q", 1:24), paste0("q", 26:28))

    missing_fields <- sapply(required_fields, function(field) input[[field]] == "")

    if (any(missing_fields)) {
      showNotification("Please fill in all required fields", type = "warning")
      return()
    }

    # Map text responses to numeric indices
    likert_to_index <- c("Strongly Agree" = 1, "Agree" = 2, "Disagree" = 3, "Strongly Disagree" = 4)
    frequency_to_index <- c("Never" = 1, "Once or Twice" = 2, "More than Twice" = 3)

    # Collect survey data with appropriate mappings for each question type
    survey_responses <- sapply(1:28, function(i) {
      if (i <= 19 || i == 28) {
        # Q1-Q19 and Q28 use Likert scale
        likert_to_index[input[[paste0("q", i)]]]
      } else if (i >= 20 && i <= 24) {
        # Q20-Q24 use frequency scale
        frequency_to_index[input[[paste0("q", i)]]]
      } else if (i == 26 || i == 27) {
        # Q26-Q27 use frequency scale
        frequency_to_index[input[[paste0("q", i)]]]
      } else {
        # Q25 is skipped (returns NA)
        NA
      }
    })
    
    # Make prediction with error handling
    prediction <- tryCatch({
      predict_school_climate(survey_responses, input$race, input$gender)
    }, error = function(e) {
      showNotification(paste("Prediction failed:", e$message), type = "error")
      return("Error in prediction")
    })
    
    # Add to data
    new_row_data <- list(
      StudentID = input$student_id,
      Prediction = prediction,
      DateTime = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
      Race = input$race,
      Gender = input$gender
    )

    # Add all questions (Q25 doesn't exist in UI, so explicitly set to empty)
    for(i in 1:28) {
      if(i == 25) {
        new_row_data[[paste0("Q", i)]] <- ""  # Q25 not used in model
      } else {
        new_row_data[[paste0("Q", i)]] <- input[[paste0("q", i)]]
      }
    }

    new_row <- data.frame(new_row_data, stringsAsFactors = FALSE)
    
    responses_data$data <- rbind(responses_data$data, new_row)
    
    showNotification("Survey submitted successfully!", type = "message")
    
    # Reset form
    updateTextInput(session, "student_id", value = "")
    updateSelectInput(session, "race", selected = "")
    updateSelectInput(session, "gender", selected = "")
    # Reset all questions (skip q25 as it's not in UI)
    lapply(c(paste0("q", 1:24), paste0("q", 26:28)), function(x) updateSelectInput(session, x, selected = ""))
  })
  
  # CSV Upload handler
  observeEvent(input$upload_csv, {
    req(input$upload_csv)

    tryCatch({
      uploaded_data <- read.csv(input$upload_csv$datapath, stringsAsFactors = FALSE)

      # Validate the uploaded data has the correct columns
      expected_cols <- c("StudentID", "Prediction", "DateTime", "Race", "Gender",
                         paste0("Q", 1:28))

      if (!all(expected_cols %in% names(uploaded_data))) {
        showNotification("CSV file does not have the correct column structure",
                         type = "error")
        return()
      }

      # Load the data
      responses_data$data <- uploaded_data
      showNotification(paste("Successfully loaded", nrow(uploaded_data), "responses"),
                       type = "message")

    }, error = function(e) {
      showNotification(paste("Error loading CSV:", e$message), type = "error")
    })
  })

  # Score card calculations
  output$total_responses <- renderText({
    nrow(responses_data$data)
  })
  
  output$high_risk_count <- renderText({
    if(nrow(responses_data$data) == 0) return("0")
    sum(responses_data$data$Prediction == "Dropout", na.rm = TRUE)
  })
  
  output$low_risk_count <- renderText({
    if(nrow(responses_data$data) == 0) return("0")
    sum(responses_data$data$Prediction == "Not_Dropout", na.rm = TRUE)
  })
  
  output$risk_percentage <- renderText({
    if(nrow(responses_data$data) == 0) return("0%")

    total_students <- nrow(responses_data$data)
    high_risk_students <- sum(responses_data$data$Prediction == "Dropout", na.rm = TRUE)

    risk_percent <- (high_risk_students / total_students) * 100
    paste0(round(risk_percent, 1), "%")
  })
  
  # Render table
  output$responses_table <- DT::renderDataTable({
    # Format the prediction column with colored chips
    data_formatted <- responses_data$data
    if(nrow(data_formatted) > 0) {
      data_formatted$Prediction <- ifelse(
        data_formatted$Prediction == "Dropout",
        '<span style="background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">Dropout</span>',
        '<span style="background-color: #28a745; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">Not Dropout</span>'
      )
    }

    DT::datatable(data_formatted,
                  options = list(pageLength = 15, scrollX = TRUE),
                  escape = FALSE)  # Allow HTML in the table
  })

  # Download handler for CSV export
  output$download_csv <- downloadHandler(
    filename = function() {
      paste("school_climate_survey_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      # Create a clean version of the data without HTML formatting
      clean_data <- responses_data$data
      if(nrow(clean_data) > 0) {
        # Clean the prediction column by removing HTML tags
        clean_data$Prediction <- gsub('<span[^>]*>|</span>', '', clean_data$Prediction)
        clean_data$Prediction <- gsub('Dropout', 'Dropout', clean_data$Prediction)
        clean_data$Prediction <- gsub('Not Dropout', 'Not_Dropout', clean_data$Prediction)
      }
      write.csv(clean_data, file, row.names = FALSE)
    }
  )

  # Download handler for template CSV
  output$download_template <- downloadHandler(
    filename = function() {
      paste("survey_template_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      # Create template data with 10 example rows
      template_data <- data.frame(
        StudentID = paste0("STU", sprintf("%04d", 1:10)),
        Prediction = c("Not_Dropout", "Dropout", "Not_Dropout", "Not_Dropout", "Dropout",
                       "Not_Dropout", "Not_Dropout", "Dropout", "Not_Dropout", "Not_Dropout"),
        DateTime = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        Race = c("White", "Black or African American", "Hispanic or Latino (No Race)",
                 "Asian or Pacific Islander", "White", "Two or More Races",
                 "White", "Hispanic or Latino (Race Specified)", "White", "Black or African American"),
        Gender = c("Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"),
        Q1 = rep("Agree", 10),
        Q2 = rep("Strongly Agree", 10),
        Q3 = rep("Agree", 10),
        Q4 = c("Disagree", "Agree", "Disagree", "Disagree", "Strongly Agree", "Disagree", "Disagree", "Agree", "Disagree", "Disagree"),
        Q5 = rep("Agree", 10),
        Q6 = rep("Agree", 10),
        Q7 = rep("Agree", 10),
        Q8 = c("Disagree", "Agree", "Disagree", "Disagree", "Agree", "Disagree", "Disagree", "Strongly Agree", "Disagree", "Disagree"),
        Q9 = c("Disagree", "Agree", "Disagree", "Disagree", "Agree", "Disagree", "Disagree", "Agree", "Disagree", "Strongly Disagree"),
        Q10 = c("Disagree", "Agree", "Disagree", "Disagree", "Strongly Agree", "Disagree", "Disagree", "Agree", "Disagree", "Disagree"),
        Q11 = c("Disagree", "Agree", "Disagree", "Disagree", "Agree", "Disagree", "Disagree", "Agree", "Disagree", "Disagree"),
        Q12 = c("Disagree", "Agree", "Disagree", "Disagree", "Agree", "Disagree", "Disagree", "Strongly Agree", "Disagree", "Disagree"),
        Q13 = c("Strongly Disagree", "Agree", "Disagree", "Strongly Disagree", "Strongly Agree", "Disagree", "Strongly Disagree", "Agree", "Disagree", "Strongly Disagree"),
        Q14 = c("Strongly Disagree", "Agree", "Disagree", "Strongly Disagree", "Agree", "Disagree", "Strongly Disagree", "Strongly Agree", "Disagree", "Strongly Disagree"),
        Q15 = rep("Agree", 10),
        Q16 = c("Agree", "Disagree", "Agree", "Agree", "Disagree", "Agree", "Agree", "Disagree", "Agree", "Agree"),
        Q17 = c("Agree", "Disagree", "Agree", "Agree", "Disagree", "Agree", "Strongly Agree", "Disagree", "Agree", "Agree"),
        Q18 = rep("Agree", 10),
        Q19 = rep("Agree", 10),
        Q20 = c("Never", "Once or Twice", "Never", "Never", "More than Twice", "Never", "Never", "Once or Twice", "Never", "Never"),
        Q21 = c("Never", "Once or Twice", "Never", "Never", "More than Twice", "Never", "Never", "More than Twice", "Never", "Never"),
        Q22 = c("Never", "Once or Twice", "Never", "Never", "More than Twice", "Never", "Never", "Once or Twice", "Never", "Never"),
        Q23 = c("Never", "Once or Twice", "Never", "Never", "More than Twice", "Never", "Never", "Once or Twice", "Never", "Never"),
        Q24 = c("Never", "Once or Twice", "Never", "Never", "More than Twice", "Never", "Never", "Once or Twice", "Never", "Never"),
        Q25 = rep("", 10),  # Q25 (BYS22F) not used in model
        Q26 = c("Never", "Once or Twice", "Never", "Never", "More than Twice", "Never", "Never", "Once or Twice", "Never", "Never"),
        Q27 = c("Never", "More than Twice", "Never", "Never", "Once or Twice", "Never", "Never", "Once or Twice", "Never", "Never"),
        Q28 = rep("Agree", 10),
        stringsAsFactors = FALSE
      )
      write.csv(template_data, file, row.names = FALSE)
    }
  )
}

shinyApp(ui = ui, server = server)