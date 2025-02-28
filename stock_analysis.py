import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

def fetch_stock_data(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    hist.reset_index(inplace=True)
    return hist

def plot_stock_price(hist, ticker):
    fig = px.line(hist, x="Date", y="Close", title=f"{ticker} Stock Price (Last 6 Months)", markers=True)
    fig.show()

def plot_candlestick(hist, ticker):
    fig = go.Figure(data=[go.Candlestick(x=hist['Date'],
                                         open=hist['Open'],
                                         high=hist['High'],
                                         low=hist['Low'],
                                         close=hist['Close'])])
    fig.update_layout(title=f"{ticker} Candlestick Chart (Last 6 Months)")
    fig.show()

def plot_moving_averages(hist, ticker):
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
    fig = px.line(hist, x='Date', y=['Close', 'SMA_20', 'EMA_20'],
                  title=f"{ticker} Moving Averages (Last 6 Months)",
                  labels={'value': 'Price (USD)', 'Date': 'Date'})
    fig.show()

def plot_volume(hist, ticker):
    fig = px.bar(hist, x='Date', y='Volume', title=f"{ticker} Trading Volume (Last 6 Months)")
    fig.show()

def analyze_stock(ticker):
    hist = fetch_stock_data(ticker)
    plot_stock_price(hist, ticker)
    plot_candlestick(hist, ticker)
    plot_moving_averages(hist, ticker)
    plot_volume(hist, ticker)

websearch_agent = Agent(name="web search agent", 
                        role="search the web", 
                        model=Groq(id="llama-3.3-70b-versatile"),
                        tools=[DuckDuckGo()], 
                        instructions=["Always include sources"], 
                        show_tool_calls=True, markdown=True)

finance_agent = Agent(name="Finance AI agent", 
                      model=Groq(id="llama-3.3-70b-versatile"),
                      tools=[YFinanceTools(stock_price=True, 
                      analyst_recommendations=True, 
                      stock_fundamentals=True, company_news=True)], 
                      instructions=["Use tables to display data"], 
                      show_tool_calls=True, markdown=True)

multi_ai_agent = Agent(team=[websearch_agent, finance_agent], 
                       model=Groq(id="llama-3.3-70b-versatile"),
                       instructions=["Always include sources", "Use tables to display data"], 
                       show_tool_calls=True, markdown=True)

ticker = input("Enter the stock ticker symbol: ").upper()

if ticker:
    multi_ai_agent.print_response(
        f"Summarize analyst recommendation and share the latest news for {ticker}",
        stream=True
    )
    analyze_stock(ticker)
else:
    print("Invalid ticker. Please enter a valid stock symbol.")


