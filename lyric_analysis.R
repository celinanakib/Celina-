# ===============================================
# Fill in the following fields
# ===============================================
# Title:
# Description: Shiny App for lyric analysis
# Author: Celina Alnakib
# Date: December 4th, 2021


# ===============================================
# Packages
# ===============================================
library(tidyverse)
library(tidytext)
library(dplyr)
library(ggplot2)
library(wordcloud)
library(shiny)
library(data.table)

# ===============================================
# Import data
# ===============================================
dat <- read.csv("u2-lyrics.csv")

# ===============================================
# Define UserInterface "ui" for application
# ===============================================

ui <- fluidPage(
  
  titlePanel("U2 Text Analysis"),
  h5("These analyses help us visualize the results from a text analysis performed on the song lyrics of Irish rock band U2"),
  fluidRow(
    
    # replace with your widgets
    
    column(3,
           p(em("Album Details")),
           selectInput(inputId = "album", 
                       label = "Album Name:",
                       choices = c(dat$album,"All Albums"),
                       selected = "All Albums")
        
    ),
    
    # replace with your widgets
    column(3,
           p(em("Words Included For Word Frequency Graph")),
           radioButtons(inputId = "stopwords", 
                        label = "Types of Words:", 
                        choices = c("With Stopwords" = "with",
                                    "Without Stopwords" = "without"),
                        selected = "with")
    ),
    # replace with your widgets
    column(3,
           p(em("Details")),
           sliderInput("numsongs",
                       "Maximum Number of Words:",
                       min = 1,  max = 100,  value = 5)
    ),
    # replace with your widgets
    column(3,
           p(em("Wordcloud Widget For Word Frequency Cloud")),
           
           sliderInput("freq",
                       "Minimum Frequency:",
                       min = 1,  max = 200, value = 15),
    )
  ),
  
  hr(),
  tabsetPanel(type = "tabs",
              tabPanel("Word Frequency Analysis",
                       h3("Word Frequency Analysis"),
                       h5("In this section, we can explore the word frequencies associated with the U2 albums ie. we can look and answer questions like what are the top-5, or top-10, or top-20 (or any other number of) most frequent words
                       used in U2 lyrics (among a given album or all albums)?"),
                       plotOutput("barplot"),
                       plotOutput("cloud1"),
                       hr(),
                       dataTableOutput('table1')),
              tabPanel("Sentiment Analysis", 
                       h3("Sentiment Analysis"),
                       h5("In this section, we can explore and visualize which words contribute to the largest positive or negative sentiment scores in every albums ie. we can look and answer questions like which words contribute the most to the “relatively large” positive or negative scores?"),
                       plotOutput("histogram"),
                       hr(),
                       verbatimTextOutput('table2'))
  )
)

# ===============================================
# Define Server "server" logic
# ===============================================

server <- function(input, output) {
  # you may need to create reactive objects
  data <- reactive({
    numsongs = input$numsongs
    if (input$album == "All Albums"){
      all_lyrics_vector <- c(dat$lyrics)
      all_lyrics_frame <- data.frame(text = all_lyrics_vector)
      tokens <- unnest_tokens(tbl = all_lyrics_frame, output = word, input = text)
    }
    else{
      album_only <- filter(dat,album == input$album)
      album_lyrics_vector <- c(album_only$lyrics)
      album_lyrics_frame <- data.frame(text = album_lyrics_vector)
      tokens <- unnest_tokens(tbl = album_lyrics_frame, output = word, input = text)
    }
    if(input$stopwords == "with"){
      to_plot <- tokens %>%
        count(word)%>%
        arrange(desc(n))%>%
        slice_head(n = numsongs)
    }
    else{
      to_plot <- tokens %>%
        anti_join(stop_words, by = "word") %>%
        count(word) %>%
        arrange(desc(n)) %>%
        slice_head(n = numsongs)
    }
    data <- to_plot
  })
  
  u2sentiment <- reactive({
    
    numsongs = input$numsongs
    if (input$album == "All Albums"){
      all_lyrics_vector <- c(dat$lyrics)
      all_lyrics_frame <- data.frame(text = all_lyrics_vector)
      tokens <- unnest_tokens(tbl = all_lyrics_frame, output = word, input = text)
    }
    
    else{
      album_only <- filter(dat,album == input$album)
      album_lyrics_vector <- c(album_only$lyrics)
      album_lyrics_frame <- data.frame(text = album_lyrics_vector)
      tokens <- unnest_tokens(tbl = album_lyrics_frame, output = word, input = text)
    }
    
    tidy_u2 <- 
      tokens %>%
      anti_join(stop_words, by = "word")
    
    u2norsentiment <-  tidy_u2  %>%
      inner_join(sentiments, by = "word" )%>%
      count(word,sentiment, sort = TRUE) %>%
      ungroup() %>%
      slice_head(n = numsongs)
    
    u2norsentiment
    
  })
  
  
  getsummarystats <- reactive({
    
    numsongs = input$numsongs
    albumname <- input$album
    
    if (input$album == "All Albums"){
      all_lyrics_vector <- c(dat$lyrics)
      all_lyrics_frame <- data.frame(text = all_lyrics_vector)
      tokens <- unnest_tokens(tbl = all_lyrics_frame, output = word, input = text)
    }
    
    else{
      album_only <- filter(dat,album == input$album)
      album_lyrics_vector <- c(album_only$lyrics)
      album_lyrics_frame <- data.frame(text = album_lyrics_vector)
      tokens <- unnest_tokens(tbl = album_lyrics_frame, output = word, input = text)
    }
    
    #logic
    tidy_u2 <- 
      tokens %>%
      anti_join(stop_words, by = "word")  %>%
      inner_join(sentiments, by = "word" )%>%
      count(word,sentiment, sort = TRUE) %>%
      slice_head(n = numsongs)
    
    return(tidy_u2)
    
  })
  
  # ===============================================
  # Outputs for the first TAB (i.e. barchart)
  # ===============================================
  
  # code for barplot
  output$barplot <- renderPlot({
    # replace the code below with your code!!!
    ggplot(data = data(), aes(x = n , y = word)) +
      geom_col() + labs(title = "Most Frequent Words") +
      ylab("word") +
      xlab("count")
  })
  
  # code for numeric summaries of frequencies
  output$table1 <- renderDataTable({
    # replace the code below with your code!!!
    data()
  })
  
  # Make the wordcloud drawing predictable during a session
  
  output$cloud1 <- renderPlot({
    wordcloud(
      words = data()$word, 
      freq = data()$n, 
      max.words = input$numsongs,
      min.freq = input$freq,
      random.order = FALSE,
      colors = brewer.pal(8, "Dark2"))
    
  })
  
  # ===============================================
  # Outputs for the second TAB (i.e. histogram)
  # ===============================================
  
  # code for histogram
  output$histogram <- renderPlot({
    # replace the code below with your code!!!
    ggplot(data = u2sentiment()) +
      geom_col(aes(x = reorder(word, n), y = n, fill = sentiment)) +
      coord_flip() +
      labs(y = "Contribution to sentiment",
           x = NULL,
           title = "Words that contribute to positive and negative sentiments") 
  })
  
  # code for statistics
  output$table2 <- renderPrint({
    # replace the code below with your code!!!
    getsummarystats()
  })
}
# ===============================================
# Run the application
# ===============================================

shinyApp(ui = ui, server = server)