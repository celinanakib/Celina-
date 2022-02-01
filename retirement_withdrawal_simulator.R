# Stat 133, Fall 2021
# Author: Celina Alnakib

# Description: Shiny app that contains a retirement withdrawal simulator.

# Inputs:
# - the initial portfolio amount, default value of 1,000,000
# - the retirement age, default value of 60 
# - the withdrawal rate, default value of 4%
# - the average annual rate of return
# - the average return volatility
# - the average inflation rate
# - the average inflation volatility
# - the number of simulations, default value of 50.
# - the value of the random seed, this is the value to be passed to set.seed().

# Details: run various simulations that give you a theoretical idea for a certain withdrawal rate.



library(dplyr)
library(ggplot2)
library(shiny)
library(reshape2)
library(DT)
library(tidyr)
library(quantreg)

# ===============================================
# Define UserInterface "ui" for application
# ===============================================


ui <- fluidPage(

    titlePanel("Retirement Withdrawal Simulator"),
    h5("This simulator will aid you to visualise and quantify the behaviour of your portfolio, which depends on the variables listed below:"),
    fluidRow(
        # Inputs for initial portfolio, retirement age, withdrawal rate, number of simulations, and random seed.
        column(3,
               br(),
               h4("Investment Variables:"),
               numericInput("initial_portfolio", 
                            label = h5("Initial Portfolio (in $)"), 
                            value = 1000000),
               numericInput("retirement_age", 
                            label = h5("Retirement Age"), 
                            value = 60),
               sliderInput("withdrawal_rate", 
                           label = h5("Withdrawal Rate (in Percent)"), 
                           min = 0, 
                           max = 30, 
                           value = 4)
        ),
        
        # Inputs for seed, type of contribution, annual return, annual volatility, random seed
        column(3,
               br(),
               h4("Interest Rate Variables:"),
               sliderInput("annual_mean_return", 
                           label = h5("Average Annual Return (in %)"), 
                           min = 0, 
                           max = 30, 
                           value = 10),
               sliderInput("annual_volatility", 
                           label = h5("Average Return Volatility (in %)"), 
                           min = 0, 
                           max = 30, 
                           value = 18)

        ),
        
        # Inputs for customizing the graph
        column(3,
               br(),
               sliderInput("annual_inflation", 
                           label = h5("Average Inflation Rate (in %)"), 
                           min = 0, 
                           max = 20, 
                           value = 3),
               sliderInput("annual_inf_volatility", 
                           label = h5("Average Inflation Volatility (in %)"), 
                           min = 0, 
                           max = 5, 
                           value = 3.5)
        ),
        
        # Inputs for data table
        
        column(3,
               br(),
        numericInput("simulations", 
                     label = h5("Number of Simulations"), 
                     value = 50),
        numericInput("seed", 
                     label = h5("Random Seed"), 
                     value = 12345)
        )
    ),
    
    hr(),
    h3('Graphical Representation of the Behaviour of the Simulator'),
    h5("This plot graphically represents the behaviour of the stimulator given the withdrawal rate"),
    br(),
    plotOutput('plot', height = 600),
    
    hr(),
    h3('Data Statistics for the Stimulator'),
    h5("This section deals with the statistics and quantitative representation of the 
     stimulator."),
    br(),
    DT::dataTableOutput("table1", width = 700),
    DT::dataTableOutput("table2", width = 700)
)
    

# ===============================================
# Define Server "server" logic
# ===============================================

#Making the data frame from the graph
server <- function(input, output) {
    #Making the dataframe from the graph


    dat <- reactive({
        
        set.seed(input$seed)
        initial.portfolio = input$initial_portfolio
        
        # Investment
        annual.mean.return = (input$annual_mean_return) / 100
        annual.volatility = (input$annual_volatility) / 100
        
        # Inflation
        annual.inflation = (input$annual_inflation) / 100
        annual.inf.volatility = (input$annual_inf_volatility) / 100
        
        # Withdrawals
        withdrawal.rate = (input$withdrawal_rate)/100
        
        # Age to consider (in Years)
        years_left = (100 - (input$retirement_age))
        
        
        # Number of simulations
        n_sim = input$simulations
        
        #-------------------------------------
        # Simulation
        #-------------------------------------
        
        
    

        
        # simulate Withdrawals
        withdrawal =  initial.portfolio * withdrawal.rate
        nav = as.list(1:input$simulations)
        balance = initial.portfolio
        
        for (s in 1:input$simulations) {
          balance = initial.portfolio
          vec <- NA
          for (i in 1:years_left) {
            invest.returns = rnorm(1 , mean = annual.mean.return, sd = annual.volatility)
            inflation.returns = rnorm(1, mean = annual.inflation, sd = annual.inf.volatility)
            balance = balance * (1 + invest.returns) - withdrawal* (1 +  inflation.returns)
            vec[[i]] = balance 
            
          }
          
         nav[[s]] = vec
            
        }
          
          table = data.frame(nav)
          names(table) = paste0("sim", 1:input$simulations)
          table$year = (1:years_left)
          
          # reshape table into "long" (or "tall") format
          pivot_longer(
            table,
            cols = starts_with("sim"),
            names_to = "simulation",
            values_to = "amount")
  
          
    })
    # code for graphs
    
    
    output$plot <- renderPlot({
      
      ggplot(dat(), aes(x = year, y = amount, group = simulation)) +
        geom_line(aes(color = simulation)) + 
        geom_point(aes(color = simulation)) +
        xlab("Years Since Retirement")+
        ylab("Portfolio Balance ($)")+
        theme_minimal()+
        geom_hline(yintercept=0, color = 'red', size = 1.5) +
        geom_quantile(aes(x = year, y = amount, group = NULL), quantiles = c(0.1,0.9), color = 'orange', method = 'rqss')+
        geom_quantile(aes(x = year, y = amount, group = NULL),quantiles = 0.5 , color = 'blue')
    })
      
      
      # code for statistics
      output$table1 <- DT::renderDataTable({
        summarize(dat(),
                  Minimum = min(amount),
                  Maximum = max(amount),
                  Median = median(amount),
                  Mean= mean(amount),
                  Minimum = min(amount),
                  "Standard Deviation" = sd(amount)
        )
      })
      
      output$table2 <- DT::renderDataTable({
        summarize(dat(),
                  "10th Percentile" = quantile(amount, probs = 0.1),
                  "25th Percentile" = quantile(amount, probs = 0.25),
                  "75th Percentile" = quantile(amount, probs = 0.75),
                  "90th Percentile" = quantile(amount, probs = 0.9)
        )
      })
      
}


    
                  
      
        

# ===============================================
# Run the application
# ===============================================
shinyApp(ui = ui, server = server)

